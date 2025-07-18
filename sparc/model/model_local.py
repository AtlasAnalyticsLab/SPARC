import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple, Dict, Optional, List

from sparc.loss import normalized_mean_squared_error
from sparc.kernels import TritonDecoderAutograd
from sparc.model.utils import unit_norm_decoder_

# --- Activation Function ---
class TopK(nn.Module):
    def __init__(self, k: int, postact_fn: Callable = nn.ReLU()) -> None:
        super().__init__()
        if k < 1:
            raise ValueError("k must be ≥ 1")
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        
        # Create mask of top-k elements
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(-1, topk.indices, True)
        
        # make all other values 0
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        
        return result, mask, topk.indices

    def extra_repr(self) -> str:
        return f"k={self.k}, postact_fn={self.postact_fn.__class__.__name__}"

# --- Multi-Stream Sparse Autoencoder Model ---
class MultiStreamSparseAutoencoder(nn.Module):
    """Multi-Stream Sparse Autoencoder with per-stream TopK activation and per-stream AuxK mechanism."""
    
    def __init__(self, d_streams: Dict[str, int], n_latents: int, k: int, auxk: int = None, auxk_threshold: float = 1e-3, dead_steps_threshold: int = 1000, use_sparse_decoder: bool = False):
        super().__init__()
        self.streams = list(d_streams.keys())
        self.n_latents = n_latents
        self.k = k
        self.auxk = auxk
        self.use_sparse_decoder = use_sparse_decoder
        self.dead_steps_threshold = dead_steps_threshold
        self.auxk_threshold = auxk_threshold 

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
            # Register buffer for per-stream dead neuron tracking
            self.register_buffer(f"stats_last_nonzero_{stream_name}", torch.zeros(n_latents, dtype=torch.long))

        # --- Shared Components ---
        self.gate = TopK(k=k) 
        
    
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

    def _encode_stream(self, stream: str, x: torch.Tensor, enc: nn.Linear, pre_b: nn.Parameter, lat_b: nn.Parameter) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Encode input for a single stream to sparse latent representation."""
        # Center input
        centered_x = x - pre_b
        
        # Pass through encoder
        logits = F.linear(centered_x, enc.weight, lat_b)
        
        # Apply top-k activation
        sparse_codes, mask, indices = self.gate(logits)

        # --- Auxiliary loss component for dead neurons (using per-stream dead neuron state) ---
        auxk_indices = None
        auxk_values = None
        if self.auxk is not None:
            # Get the specific stats buffer for this stream
            stream_stats = getattr(self, f"stats_last_nonzero_{stream}")
            
            # Apply mask to focus on dead neurons (using stream-specific stats_last_nonzero)
            masked_logits = self.auxk_mask_fn(logits, stream_stats)
            
            # Get top-k values and indices from masked logits for this stream using torch.topk directly
            topk = torch.topk(masked_logits, k=self.auxk, dim=-1)
            auxk_indices = topk.indices
            auxk_values = topk.values
            
        return logits, sparse_codes, mask, indices, auxk_indices, auxk_values

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

    def _encode_all_streams(self, inputs: Dict[str, torch.Tensor], stream_keys: set = None) -> Dict[str, torch.Tensor]:
        """Encode all input streams and update per-stream dead neuron stats."""
        if stream_keys is None:
            stream_keys = set(self.streams)
            
        outputs = {}
        for stream in stream_keys:
            enc, _, pre_b, lat_b = self._get_stream_params(stream)
            logits, sparse_codes, mask, indices, auxk_indices, auxk_values = self._encode_stream(
                stream, inputs[stream], enc, pre_b, lat_b # Pass stream name
            )
            outputs[f'logits_{stream}'] = logits
            outputs[f'sparse_codes_{stream}'] = sparse_codes
            outputs[f'mask_{stream}'] = mask
            outputs[f'indices_{stream}'] = indices

            # --- Update Per-Stream Dead Neuron Stats ---
            # Resets counter if sparse_codes > auxk_threshold, otherwise increments.
            stream_stats = getattr(self, f"stats_last_nonzero_{stream}")
            
            activated_mask_batch = (sparse_codes > self.auxk_threshold) # Shape [batch, n_latents]
            activated_mask_latents = activated_mask_batch.any(dim=0).to(stream_stats.dtype) # Shape [n_latents]

            stream_stats *= (1 - activated_mask_latents) # Reset counter to 0 if activated
            stream_stats += 1 # Increment counter for all (activated now become 1, non-activated increase)

            if auxk_indices is not None:
                outputs[f'auxk_indices_{stream}'] = auxk_indices
                outputs[f'auxk_values_{stream}'] = auxk_values
                
        return outputs

    def _decode_self_reconstructions(self, outputs: Dict[str, torch.Tensor], stream_keys: set = None) -> Dict[str, torch.Tensor]:
        """Decode self-reconstructions for each stream and calculate average active neurons/sparsity."""
        if stream_keys is None:
            stream_keys = set(self.streams)
            
        total_sparsity = 0
        # --- Calculate Per-Stream Active Neurons & Sparsity --- 
        total_active_neurons = 0
        num_streams_with_mask = 0
        for stream in stream_keys:
            _, dec, pre_b, _ = self._get_stream_params(stream)
            recon = self._decode_stream(
                outputs[f'sparse_codes_{stream}'], 
                dec, 
                pre_b, 
                indices=outputs.get(f'indices_{stream}') if self.use_sparse_decoder else None # Use .get for safety
            )
            outputs[f'recon_{stream}'] = recon
            # Ensure mask exists before calculating sparsity
            if f'mask_{stream}' in outputs:
                total_sparsity += outputs[f'mask_{stream}'].float().sum(-1).mean()
                # Accumulate active neurons count per stream
                total_active_neurons += outputs[f'mask_{stream}'].float().sum(-1).mean()
                num_streams_with_mask += 1

        if len(stream_keys) > 0:
             outputs['avg_sparsity'] = total_sparsity / len(stream_keys)
        else:
             outputs['avg_sparsity'] = torch.tensor(0.0, device=next(self.parameters()).device)
             
        # Store the correct average number of active neurons
        if num_streams_with_mask > 0:
             outputs['avg_num_active_neurons'] = total_active_neurons / num_streams_with_mask
        else:
             outputs['avg_num_active_neurons'] = torch.tensor(0.0, device=next(self.parameters()).device)
             
        return outputs

    def _decode_cross_reconstructions(self, outputs: Dict[str, torch.Tensor], stream_keys: set = None) -> Dict[str, torch.Tensor]:
        """Decode cross-reconstructions between streams."""
        if stream_keys is None:
            stream_keys = set(self.streams)
            
        for source_stream in stream_keys:
            for target_stream in stream_keys:
                if source_stream == target_stream: continue 

                source_sparse_codes = outputs[f'sparse_codes_{source_stream}']
                _, target_dec, target_pre_b, _ = self._get_stream_params(target_stream)

                cross_recon = self._decode_stream(
                    source_sparse_codes, 
                    target_dec, 
                    target_pre_b, 
                    indices=outputs.get(f'indices_{source_stream}') if self.use_sparse_decoder else None # Use .get for safety
                )
                outputs[f'cross_recon_{target_stream}_from_{source_stream}'] = cross_recon
        return outputs

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass using per-stream TopK activation."""
        
        stream_keys = set(self.streams)
        input_keys = set(inputs.keys())
        # Validate input keys match initialized streams
        if stream_keys != input_keys:
            assert input_keys.issubset(stream_keys), f"Input keys {list(inputs.keys())} do not match initialized streams {self.streams}"
            stream_keys = input_keys
            
        # --- Encode all streams & update per-stream dead neuron stats ---
        outputs = self._encode_all_streams(inputs, stream_keys)
        
        # --- Decode self-reconstructions --- 
        outputs = self._decode_self_reconstructions(outputs, stream_keys)
            
        # --- Decode cross-reconstructions --- 
        outputs = self._decode_cross_reconstructions(outputs, stream_keys)
            
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

    def _compute_stream_losses(self, outputs: Dict[str, torch.Tensor], inputs: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], float, float, float]:
        """Compute MSE, L1 Sparsity, and AuxK loss for each stream."""
        losses = {}
        total_mse_loss = 0
        total_auxk_loss = 0
        total_l1_sparsity = 0
        num_streams = len(self.streams)
        epsilon = 1e-8 # For safe division in sparsity calculation

        for stream in self.streams:
            # --- Calculate MSE loss per stream --- 
            recon_key = f'recon_{stream}'
            if recon_key not in outputs:
                print(f"Warning: Missing reconstruction key {recon_key} in outputs.")
                continue
                
            mse_loss = normalized_mean_squared_error(outputs[recon_key], inputs[stream])
            total_mse_loss += mse_loss
            losses[f'mse_loss_{stream}'] = mse_loss
            
            # --- Calculate Per-Stream Active Neurons and L1 Sparsity --- 
            num_active_batch = torch.tensor(0.0, device=inputs[stream].device) # Default
            if f'mask_{stream}' in outputs:
                mask = outputs[f'mask_{stream}']
                num_active_batch = mask.float().sum(dim=-1) # Shape: [batch_size]
                losses[f'num_active_neurons_{stream}'] = num_active_batch.mean()
                
            # L1 Sparsity (L1 / L0 of activated sparse codes)
            if f'sparse_codes_{stream}' in outputs:
                sparse_codes = outputs[f'sparse_codes_{stream}']
                l1_norm_batch = sparse_codes.abs().sum(dim=-1) # Shape: [batch_size]
                # Calculate sparsity per batch item, handle division by zero
                l1_sparsity_batch = l1_norm_batch / (num_active_batch + epsilon) 
                avg_l1_sparsity = l1_sparsity_batch.mean()
                losses[f'l1_sparsity_{stream}'] = avg_l1_sparsity
                total_l1_sparsity += avg_l1_sparsity 
            else:
                # Default if sparse codes not found
                losses[f'l1_sparsity_{stream}'] = torch.tensor(0.0, device=inputs[stream].device)
            
            # --- Calculate AuxK loss per stream --- 
            if self.auxk is not None and f'auxk_indices_{stream}' in outputs and f'auxk_values_{stream}' in outputs:
                _, dec, pre_b, _ = self._get_stream_params(stream)
                
                # Apply ReLU to auxk values for this stream
                auxk_values = torch.relu(outputs[f'auxk_values_{stream}'])
                auxk_indices = outputs[f'auxk_indices_{stream}']
                
                # Create sparse tensor for this stream's auxk activations
                auxk_sparse = torch.zeros_like(outputs[f'sparse_codes_{stream}'])
                batch_indices = torch.arange(auxk_values.size(0), device=auxk_values.device).unsqueeze(1).expand_as(auxk_indices)
                # Ensure indices are within bounds before scattering
                valid_mask = (auxk_indices >= 0) & (auxk_indices < auxk_sparse.size(1))
                valid_batch_indices = batch_indices[valid_mask]
                valid_auxk_indices = auxk_indices[valid_mask]
                valid_auxk_values = auxk_values[valid_mask]
                
                if valid_batch_indices.numel() > 0: # Proceed only if there are valid indices
                    auxk_sparse.index_put_((valid_batch_indices, valid_auxk_indices), valid_auxk_values)
                
                # Compute auxk reconstruction for this stream
                auxk_recon = self._decode_stream(
                    auxk_sparse, # Always pass the sparse tensor representation
                    dec, 
                    pre_b, 
                    indices=auxk_indices if self.use_sparse_decoder else None
                )

                # Target for auxiliary loss is the residual error for this stream
                # Detach recon and pre_b to avoid gradients flowing back through them to the main loss
                auxk_target = inputs[stream] - outputs[recon_key].detach() + pre_b.detach()
                
                # Auxiliary loss (normalized MSE on residual) for this stream
                auxk_loss = normalized_mean_squared_error(auxk_recon, auxk_target)
                total_auxk_loss += auxk_loss
                losses[f'auxk_loss_{stream}'] = auxk_loss
                
        return losses, total_mse_loss, total_auxk_loss, total_l1_sparsity

    def _compute_cross_stream_losses(self, outputs: Dict[str, torch.Tensor], inputs: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], float, int]:
        """Compute cross-stream reconstruction losses."""
        cross_losses = {}
        total_cross_loss = 0
        num_cross_terms = 0

        for source_stream in self.streams:
            for target_stream in self.streams:
                if source_stream == target_stream: continue
                
                cross_recon_key = f'cross_recon_{target_stream}_from_{source_stream}'
                if cross_recon_key in outputs:
                    cross_loss = normalized_mean_squared_error(outputs[cross_recon_key], inputs[target_stream])
                    cross_losses[f'cross_loss_{target_stream}_from_{source_stream}'] = cross_loss
                    total_cross_loss += cross_loss
                    num_cross_terms += 1
                else:
                    print(f"Warning: Missing cross-reconstruction key {cross_recon_key} in outputs.")
        
        return cross_losses, total_cross_loss, num_cross_terms

    def _aggregate_and_finalize_losses(
        self, 
        losses: Dict[str, torch.Tensor], 
        total_mse_loss: float, 
        total_auxk_loss: float, 
        total_l1_sparsity: float,
        total_cross_loss: float, 
        num_cross_terms: int,
        outputs: Dict[str, torch.Tensor], 
        auxk_coef: float,
        cross_loss_coef: float
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Aggregate per-stream losses, apply coefficients, and compute final total loss."""
        num_streams = len(self.streams)
        device = next(self.parameters()).device 

        if num_streams > 0:
            avg_mse_loss = total_mse_loss / num_streams
            losses["avg_mse_loss"] = avg_mse_loss
            losses["avg_num_active_neurons"] = outputs.get('avg_num_active_neurons', torch.tensor(0.0, device=device))
            losses["avg_l1_sparsity"] = total_l1_sparsity / num_streams
        else:
            avg_mse_loss = torch.tensor(0.0, device=device)
            losses["avg_mse_loss"] = avg_mse_loss
            losses["avg_num_active_neurons"] = torch.tensor(0.0, device=device)
            losses["avg_l1_sparsity"] = torch.tensor(0.0, device=device)
            
        total_loss = avg_mse_loss

        # Add AuxK loss contribution if calculated
        if self.auxk is not None and total_auxk_loss > 0 and num_streams > 0:
            avg_auxk_loss = total_auxk_loss / num_streams
            avg_auxk_loss_scaled = avg_auxk_loss * auxk_coef
            losses["avg_auxk_loss"] = avg_auxk_loss
            losses["avg_auxk_loss_scaled"] = avg_auxk_loss_scaled
            total_loss += avg_auxk_loss_scaled
        elif self.auxk is not None:
            losses["avg_auxk_loss"] = torch.tensor(0.0, device=device)
            losses["avg_auxk_loss_scaled"] = torch.tensor(0.0, device=device)
        
        # Add Cross-Stream loss contribution if calculated
        if num_cross_terms > 0 and cross_loss_coef > 0:
             avg_cross_loss = total_cross_loss / num_cross_terms
             avg_cross_loss_scaled = avg_cross_loss * cross_loss_coef
             losses["avg_cross_loss"] = avg_cross_loss
             losses["avg_cross_loss_scaled"] = avg_cross_loss_scaled
             total_loss += avg_cross_loss_scaled
        elif cross_loss_coef > 0:
             losses["avg_cross_loss"] = torch.tensor(0.0, device=device)
             losses["avg_cross_loss_scaled"] = torch.tensor(0.0, device=device)
             
        losses["total_loss"] = total_loss
        
        return total_loss, losses

    def compute_loss(self, outputs: Dict[str, torch.Tensor], inputs: Dict[str, torch.Tensor], auxk_coef: float = 1/32, cross_loss_coef: float = 1.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss including per-stream MSE, AuxK, L1 Sparsity, and Cross-Stream terms.
        
        Args:
            outputs: Output dictionary from forward pass.
            inputs: Dictionary containing input tensors (keys must match initialized streams).
            auxk_coef: Weight for auxiliary loss.
            cross_loss_coef: Weight for cross-stream reconstruction loss.
            
        Returns:
            Tuple: (Total loss tensor, Dictionary of loss components)
        """
        # Validate input keys match initialized streams
        if set(inputs.keys()) != set(self.streams):
            raise ValueError(f"Input keys {list(inputs.keys())} do not match initialized streams {self.streams}")

        # --- Compute per-stream losses (MSE, L1, AuxK) ---
        stream_losses, total_mse_loss, total_auxk_loss, total_l1_sparsity = \
            self._compute_stream_losses(outputs, inputs)

        # --- Compute cross-stream reconstruction losses ---
        cross_losses, total_cross_loss, num_cross_terms = \
            self._compute_cross_stream_losses(outputs, inputs)

        # --- Aggregate and finalize all losses ---
        all_losses = {**stream_losses, **cross_losses} 
        total_loss, final_losses = self._aggregate_and_finalize_losses(
            losses=all_losses, 
            total_mse_loss=total_mse_loss, 
            total_auxk_loss=total_auxk_loss, 
            total_l1_sparsity=total_l1_sparsity,
            total_cross_loss=total_cross_loss, 
            num_cross_terms=num_cross_terms,
            outputs=outputs, 
            auxk_coef=auxk_coef,
            cross_loss_coef=cross_loss_coef
        )
        
        return total_loss, final_losses 