import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple, Dict, Optional, List

from sparc.loss import normalized_mean_squared_error
from sparc.kernels import TritonDecoderAutograd
from sparc.model.utils import unit_norm_decoder_

# --- Activation Function ---
class TopK(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__()
        if k < 1:
            raise ValueError("k must be ≥ 1")
        self.k = k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        topk = torch.topk(x, k=self.k, dim=-1)
        
        # Create mask of top-k elements
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(-1, topk.indices, True)
        
        return mask, topk.indices

    def extra_repr(self) -> str:
        return f"k={self.k}"

# --- Multi-Stream Sparse Autoencoder Model ---
class MultiStreamSparseAutoencoder(nn.Module):
    """Multi-Stream Sparse Autoencoder with global TopK activation on aggregated logits and per-stream AuxK mechanism."""
    
    def __init__(self, d_streams: Dict[str, int], n_latents: int, k: int, auxk: int = None, auxk_threshold: float = 1e-3, dead_steps_threshold: int = 1000, use_sparse_decoder: bool = False):
        super().__init__()
        self.streams = list(d_streams.keys())
        self.n_latents = n_latents
        self.k = k
        self.auxk = auxk 
        self.use_sparse_decoder = use_sparse_decoder
        self.dead_steps_threshold = dead_steps_threshold

        # --- Create encoder/decoder blocks for each stream ---
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.pre_biases = nn.ParameterDict()
        self.latent_biases = nn.ParameterDict()

        for stream_name, d_in in d_streams.items():
            enc = nn.Linear(d_in, n_latents, bias=False)
            dec = nn.Linear(n_latents, d_in, bias=False)
            # Initialize with tied weights & apply unit norm
            dec.weight.data = enc.weight.data.T.clone()
            unit_norm_decoder_(dec)
            
            self.encoders[stream_name] = enc
            self.decoders[stream_name] = dec
            self.pre_biases[stream_name] = nn.Parameter(torch.zeros(d_in))
            self.latent_biases[stream_name] = nn.Parameter(torch.zeros(n_latents))
        

        # --- Shared/Configuration Components ---
        self.gate = TopK(k=k) # Shared gate for global TopK index selection
        self.postact_fn = nn.ReLU() # Activation function applied *after* selection
        self.auxk_gate = TopK(k=auxk) if auxk is not None else None # Separate gate for AuxK index selection
        self.auxk_threshold = auxk_threshold # Activation threshold for dead neuron reset
        
        # Dead neuron tracking per stream
        for stream_name in self.streams:
            # Register buffer for per-stream dead neuron tracking
            self.register_buffer(f"stats_last_nonzero_{stream_name}", torch.zeros(n_latents, dtype=torch.long)) 
    
    def _get_stream_params(self, stream: str) -> Tuple[nn.Linear, nn.Linear, nn.Parameter, nn.Parameter]:
        """Helper to get parameters for a specific stream."""
        if stream in self.streams:
            return self.encoders[stream], self.decoders[stream], self.pre_biases[stream], self.latent_biases[stream]
        else:
            raise ValueError(f"Unknown stream: {stream}")

    def auxk_mask_fn(self, x: torch.Tensor, stream_stats_last_nonzero: torch.Tensor) -> torch.Tensor:
        """Apply mask to prioritize dead neurons for auxk mechanism for a specific stream."""
        dead_mask = stream_stats_last_nonzero > self.dead_steps_threshold
        x = x.clone()  # Clone to avoid modifying the original tensor
        x *= dead_mask  # Mask to keep only dead neurons' activations
        return x

    def _encode_stream(self, x: torch.Tensor, enc: nn.Linear, pre_b: nn.Parameter, lat_b: nn.Parameter) -> torch.Tensor:
        """Encode input for a single stream, returning raw logits."""
        # Center input
        centered_x = x - pre_b
        
        # Pass through encoder
        logits = F.linear(centered_x, enc.weight, lat_b)
            
        return logits

    def _decode_stream(self, sparse_codes: torch.Tensor, dec: nn.Linear, pre_b: nn.Parameter, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode sparse codes for a single stream to reconstruction.
        
        If use_sparse_decoder is True and indices are provided, uses a sparse matmul kernel.
        Otherwise, uses a standard dense matmul.
        """
        if self.use_sparse_decoder and indices is not None:
            # Gather the non-zero values corresponding to the indices for the sparse kernel
            batch_indices = torch.arange(sparse_codes.size(0), device=sparse_codes.device).unsqueeze(1).expand_as(indices)
            values = sparse_codes[batch_indices, indices] # Shape: [batch_size, k]
            
            # Use sparse decoding (Triton kernel via autograd function)
            recons = TritonDecoderAutograd.apply(indices, values, dec.weight)
            return recons + pre_b
        else:
            # Use standard dense decoding
            recon = F.linear(sparse_codes, dec.weight) + pre_b
            return recon

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass using global TopK activation on aggregated logits."""
        
        stream_keys = set(self.streams)
        input_keys = set(inputs.keys())
        # Validate input keys match initialized streams
        if stream_keys != input_keys:
            assert input_keys.issubset(stream_keys), f"Input keys {list(inputs.keys())} do not match initialized streams {self.streams}"
            stream_keys = input_keys
        outputs = {}
        all_logits = {}

        # --- Encode each stream to get raw logits --- 
        for stream in stream_keys:
            enc, _, pre_b, lat_b = self._get_stream_params(stream)
            logits = self._encode_stream(
                inputs[stream], enc, pre_b, lat_b
            )
            all_logits[stream] = logits # Store raw logits per stream
            outputs[f'logits_{stream}'] = logits 
            
        # --- Aggregate logits (summing) and apply shared TopK for INDEX SELECTION ONLY --- 
        logits_list = list(all_logits.values())
        aggregated_logits = torch.stack(logits_list, dim=0).sum(dim=0)
        
        # Get global top-k mask and indices from aggregated logits.
        shared_mask, shared_indices = self.gate(aggregated_logits)

        outputs['shared_mask'] = shared_mask
        outputs['shared_indices'] = shared_indices
                
        # --- Calculate sparse codes, update dead neurons, compute AuxK, and decode (Self-Reconstruction) --- 
        total_active_neurons = 0 # For overall avg_num_active_neurons
        for stream in stream_keys:
            logits_stream = all_logits[stream]
            sparse_codes_stream = torch.zeros_like(logits_stream)
            
            batch_indices_scatter = torch.arange(logits_stream.size(0), device=logits_stream.device).unsqueeze(1).expand_as(shared_indices)
            values_at_shared_indices = logits_stream[batch_indices_scatter, shared_indices]
            
            # Apply post-activation HERE to the stream-specific values
            activated_values = self.postact_fn(values_at_shared_indices)
            
            # Scatter activated values back
            sparse_codes_stream.scatter_(-1, shared_indices, activated_values)
            
            outputs[f'sparse_codes_{stream}'] = sparse_codes_stream 

            # Store the mask based on selected indices for consistency with TopK output format
            mask_stream = torch.zeros_like(logits_stream, dtype=torch.bool)
            mask_stream.scatter_(-1, shared_indices, True)
            outputs[f'mask_{stream}'] = mask_stream
            total_active_neurons += mask_stream.float().sum(-1).mean() # Accumulate for avg

            # --- Update Per-Stream Dead Neuron Stats ---
            # Resets counter if sparse_codes_stream > auxk_threshold, otherwise increments.
            stream_stats = getattr(self, f"stats_last_nonzero_{stream}")
            activated_mask_batch = (sparse_codes_stream > self.auxk_threshold)
            activated_mask_latents = activated_mask_batch.any(dim=0).to(stream_stats.dtype)
            stream_stats *= (1 - activated_mask_latents) # Reset if activated above threshold
            stream_stats += 1 # Increment all (active neurons become 1, inactive neurons increase count)
            
            # --- Per-Stream AuxK Calculation --- 
            if self.auxk is not None and self.auxk_gate is not None:
                masked_logits = self.auxk_mask_fn(logits_stream, stream_stats)
                auxk_mask, auxk_indices = self.auxk_gate(masked_logits)
                
                # Get values at auxk_indices from *masked* logits
                auxk_batch_indices = torch.arange(masked_logits.size(0), device=masked_logits.device).unsqueeze(1).expand_as(auxk_indices)
                auxk_values_at_indices = masked_logits[auxk_batch_indices, auxk_indices]
                
                # Apply activation
                activated_auxk_values = self.postact_fn(auxk_values_at_indices)
                
                # Create sparse tensor for auxk activations
                auxk_sparse_codes_stream = torch.zeros_like(logits_stream)
                # Ensure indices are valid before scattering
                valid_auxk_mask = (auxk_indices >= 0) & (auxk_indices < auxk_sparse_codes_stream.size(1))
                valid_auxk_batch_indices = auxk_batch_indices[valid_auxk_mask]
                valid_auxk_indices = auxk_indices[valid_auxk_mask]
                valid_activated_auxk_values = activated_auxk_values[valid_auxk_mask]
                
                if valid_auxk_batch_indices.numel() > 0:
                    auxk_sparse_codes_stream.index_put_((valid_auxk_batch_indices, valid_auxk_indices), valid_activated_auxk_values)
                    
                outputs[f'auxk_sparse_codes_{stream}'] = auxk_sparse_codes_stream
                outputs[f'auxk_indices_{stream}'] = auxk_indices # Store indices too

            # Decode using the stream-specific activated sparse codes and shared indices
            _, dec, pre_b, _ = self._get_stream_params(stream)
            recon = self._decode_stream(
                sparse_codes_stream, 
                dec, 
                pre_b, 
                indices=shared_indices if self.use_sparse_decoder else None
            )
            outputs[f'recon_{stream}'] = recon
            
        # Calculate avg num active neurons across streams
        if stream_keys:
            outputs['avg_num_active_neurons'] = total_active_neurons / len(stream_keys)
        else:
            outputs['avg_num_active_neurons'] = torch.tensor(0.0, device=activated_values.device if 'activated_values' in locals() else 'cpu')
            
        # --- Cross Reconstructions using GLOBAL indices and ACTIVATED SOURCE-stream values --- 
        for source_stream in stream_keys:
            source_sparse_codes = outputs[f'sparse_codes_{source_stream}'] # Already activated
            for target_stream in stream_keys:
                if source_stream == target_stream: continue
                
                _, target_dec, target_pre_b, _ = self._get_stream_params(target_stream)
                cross_recon = self._decode_stream(
                    source_sparse_codes, 
                    target_dec, 
                    target_pre_b, 
                    indices=shared_indices if self.use_sparse_decoder else None
                )
                outputs[f'cross_recon_{target_stream}_from_{source_stream}'] = cross_recon 
            
        return outputs
    
    def get_dead_neurons(self, threshold: int = None) -> Dict[str, torch.Tensor]:
        """Get indices of dead neurons for each stream (not activated for a long time).
        
        Args:
            threshold: Number of steps to consider a neuron dead (default: use class threshold)
            
        Returns:
            Dictionary where keys are stream names and values are boolean masks 
            of dead neurons for that stream, shape [n_latents]
        """
        if threshold is None:
            threshold = self.dead_steps_threshold

        dead_neurons_dict = {}
        for stream in self.streams:
            stream_stats = getattr(self, f"stats_last_nonzero_{stream}")
            dead_neurons_dict[stream] = stream_stats > threshold
            
        return dead_neurons_dict

    def compute_loss(self, outputs: Dict[str, torch.Tensor], inputs: Dict[str, torch.Tensor], auxk_coef: float = 1/32, cross_loss_coef: float = 1.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss including per-stream MSE, AuxK, L1 Sparsity, and Cross-Stream terms."""
        # Validate input keys match initialized streams
        if set(inputs.keys()) != set(self.streams):
            raise ValueError(f"Input keys {list(inputs.keys())} do not match initialized streams {self.streams}")

        total_mse_loss = 0
        total_auxk_loss = 0
        total_cross_loss = 0
        losses = {}
        num_streams = len(self.streams)
        num_cross_terms = 0
        # Initialize accumulators for overall averages
        total_active_neurons = 0
        total_l1_sparsity = 0
        epsilon = 1e-8 # For safe division in sparsity calculation

        # --- Calculate per-stream losses --- 
        for stream in self.streams:
            recon_key = f'recon_{stream}'
            mask_key = f'mask_{stream}'
            sparse_codes_key = f'sparse_codes_{stream}'
            
            # Calculate MSE Loss
            if recon_key in outputs and stream in inputs:
                mse_loss = normalized_mean_squared_error(outputs[recon_key], inputs[stream])
                total_mse_loss += mse_loss
                losses[f'mse_loss_{stream}'] = mse_loss
            
            # Calculate Per-Stream Active Neurons and L1 Sparsity
            if mask_key in outputs and sparse_codes_key in outputs:
                mask = outputs[mask_key]
                sparse_codes = outputs[sparse_codes_key]
                
                # Num Active Neurons (L0 norm averaged over batch)
                num_active_batch = mask.float().sum(dim=-1) # Shape: [batch_size]
                avg_num_active = num_active_batch.mean()
                losses[f'num_active_neurons_{stream}'] = avg_num_active
                total_active_neurons += avg_num_active # Accumulate for overall average
                
                # L1 Sparsity (L1 / L0, averaged over batch)
                l1_norm_batch = sparse_codes.abs().sum(dim=-1) # Shape: [batch_size]
                # Calculate sparsity per batch item, handle division by zero
                l1_sparsity_batch = l1_norm_batch / (num_active_batch + epsilon) 
                avg_l1_sparsity = l1_sparsity_batch.mean()
                losses[f'l1_sparsity_{stream}'] = avg_l1_sparsity
                total_l1_sparsity += avg_l1_sparsity # Accumulate for overall average
            else:
                # Default if mask/sparse codes not found (should not happen in normal flow)
                losses[f'num_active_neurons_{stream}'] = torch.tensor(0.0, device=outputs.get(recon_key, torch.tensor(0.)).device)
                losses[f'l1_sparsity_{stream}'] = torch.tensor(0.0, device=outputs.get(recon_key, torch.tensor(0.)).device)

            # --- Calculate AuxK loss per stream --- 
            auxk_sparse_key = f'auxk_sparse_codes_{stream}'
            auxk_indices_key = f'auxk_indices_{stream}'
            if self.auxk is not None and auxk_sparse_key in outputs and recon_key in outputs:
                auxk_sparse_codes = outputs[auxk_sparse_key]
                auxk_indices = outputs.get(auxk_indices_key) # Optional indices for sparse decoder
                
                _, dec, pre_b, _ = self._get_stream_params(stream)
                
                # Decode auxk sparse codes
                auxk_recon = self._decode_stream(
                    auxk_sparse_codes, 
                    dec, 
                    pre_b, 
                    indices=auxk_indices if self.use_sparse_decoder else None
                )
                
                # Target is residual
                auxk_target = inputs[stream] - outputs[recon_key].detach() + pre_b.detach()
                
                auxk_loss = normalized_mean_squared_error(auxk_recon, auxk_target)
                total_auxk_loss += auxk_loss
                losses[f'auxk_loss_{stream}'] = auxk_loss

        # --- Calculate Cross-Stream Reconstruction Loss --- 
        for source_stream in self.streams:
            for target_stream in self.streams:
                if source_stream == target_stream: continue
                cross_recon_key = f'cross_recon_{target_stream}_from_{source_stream}'
                if cross_recon_key in outputs:
                    cross_loss = normalized_mean_squared_error(outputs[cross_recon_key], inputs[target_stream])
                    losses[f'cross_loss_{target_stream}_from_{source_stream}'] = cross_loss
                    total_cross_loss += cross_loss
                    num_cross_terms += 1
                # else: (optional warning)

        # --- Aggregate and finalize losses --- 
        avg_mse_loss = total_mse_loss / num_streams if num_streams > 0 else torch.tensor(0.0, device=total_mse_loss.device if torch.is_tensor(total_mse_loss) else 'cpu')
        losses["avg_mse_loss"] = avg_mse_loss
        # Use the overall average active neurons calculated in forward pass
        losses["avg_num_active_neurons"] = outputs.get('avg_num_active_neurons', torch.tensor(0.0))
        # Calculate and store average L1 sparsity across streams
        if num_streams > 0:
            losses["avg_l1_sparsity"] = total_l1_sparsity / num_streams
        else:
            losses["avg_l1_sparsity"] = torch.tensor(0.0, device=outputs.get(recon_key, torch.tensor(0.)).device if 'recon_key' in locals() else 'cpu')
        
        total_loss = avg_mse_loss

        # Add AuxK loss contribution if calculated
        if self.auxk is not None and total_auxk_loss > 0 and num_streams > 0:
            avg_auxk_loss = total_auxk_loss / num_streams
            avg_auxk_loss_scaled = avg_auxk_loss * auxk_coef
            losses["avg_auxk_loss"] = avg_auxk_loss
            losses["avg_auxk_loss_scaled"] = avg_auxk_loss_scaled
            total_loss += avg_auxk_loss_scaled
        elif self.auxk is not None:
            # Ensure keys exist even if loss is zero
            losses["avg_auxk_loss"] = torch.tensor(0.0, device=total_loss.device)
            losses["avg_auxk_loss_scaled"] = torch.tensor(0.0, device=total_loss.device)

        # Add Cross-Stream loss contribution if calculated
        if num_cross_terms > 0 and cross_loss_coef > 0:
             avg_cross_loss = total_cross_loss / num_cross_terms
             avg_cross_loss_scaled = avg_cross_loss * cross_loss_coef
             losses["avg_cross_loss"] = avg_cross_loss
             losses["avg_cross_loss_scaled"] = avg_cross_loss_scaled
             total_loss += avg_cross_loss_scaled
             
        losses["total_loss"] = total_loss
        
        return total_loss, losses  