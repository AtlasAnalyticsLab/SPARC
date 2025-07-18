import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import wandb

from sparc.model.utils import unit_norm_decoder_, unit_norm_decoder_grad_adjustment_

# --- Training Function ---
def train_loop(
    model,
    dataloader: DataLoader, # Expects dataloader to yield Dict[str, np.ndarray]
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 10,
    dead_neuron_threshold: int = 1000,
    auxk_coef: float = 1/32,
    cross_loss_coef: float = 1.0,
    wandb_run = None,
    val_dataloader = None
):
    """Train the Multi-Stream Sparse Autoencoder, tracking metrics per stream.
    
    Args:
        model: MultiStreamSparseAutoencoder model (with global or local topk)
        dataloader: Data loader yielding dictionaries {stream_name: feature_tensor}
        optimizer: Optimizer
        device: Device to use
        num_epochs: Number of epochs
        dead_neuron_threshold: Steps threshold to consider a neuron dead
        auxk_coef: Weight for auxiliary loss
        cross_loss_coef: Weight for cross-loss
        wandb_run: Optional wandb run object for logging.
        val_dataloader: Optional validation data loader for tracking validation loss
    """
    model.train()
    global_step = 0
    
    # Training metrics - Dictionaries to track per stream
    epoch_total_losses = []
    recon_losses = defaultdict(list)
    active_neurons = defaultdict(list) 
    l1_sparsities = defaultdict(list) 
    auxk_losses = defaultdict(list) 
    cross_losses = defaultdict(list) 
    
    # Validation metrics
    val_epoch_total_losses = []
    val_recon_losses = defaultdict(list)
    val_active_neurons = defaultdict(list)
    val_l1_sparsities = defaultdict(list)
    val_auxk_losses = defaultdict(list)
    val_cross_losses = defaultdict(list)
    
    # Dead neuron tracking
    epoch_dead_neurons = defaultdict(list)
    
    for epoch in range(num_epochs):
        epoch_loss_sum = 0.0 
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        # Reset batch metrics for epoch average calculation
        batch_recon_losses = defaultdict(list)
        batch_active_neurons = defaultdict(list)
        batch_auxk_losses = defaultdict(list)
        batch_cross_losses = defaultdict(list) 
        batch_l1_sparsities = defaultdict(list) 
             
        for batch_idx, batch_data in enumerate(progress_bar):
            # Expecting batch_data to be a dict: {stream_name: np_array, ...}
            if not isinstance(batch_data, dict):
                raise ValueError(f"Dataloader expected to yield dict, but got {type(batch_data)}")
            inputs = {}
            for stream_name, data_np in batch_data.items():        
                if not isinstance(data_np, np.ndarray):
                    try:
                        data_np = np.array(data_np)
                    except Exception as e:
                        raise TypeError(f"Could not convert batch data for stream '{stream_name}' to NumPy array: {e}")
                        
                inputs[stream_name] = torch.from_numpy(data_np).to(device, non_blocking=True)
            
            if not inputs:
                print(f"Warning: Skipping batch {batch_idx} due to no matching streams found in data.")
                continue
                
            optimizer.zero_grad()
            output = model(inputs)
            
            # Compute loss
            loss, metrics = model.compute_loss(output, inputs, auxk_coef, cross_loss_coef)
            
            # Backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf loss detected at step {global_step}. Skipping batch.")
                print("Metrics:", metrics)
                continue
                
            loss.backward()
            
            # Clip gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Iterate over streams for decoder updates per stream
            for stream in model.streams:
                 if stream in model.decoders:
                     dec = model.decoders[stream]
                     # Re-normalize decoder weights
                     unit_norm_decoder_(dec)
                     # Apply gradient adjustment
                     unit_norm_decoder_grad_adjustment_(dec)
            
            optimizer.step()
            
            # --- Logging to Wandb (if enabled) ---
            if wandb_run:
                log_data = {
                    "step": global_step,
                    "epoch": epoch,
                    "total_loss": loss.item(),
                    "num_active_neurons": metrics.get('avg_num_active_neurons', torch.tensor(0.0)).item(), # Use the average number of active neurons
                    "sparsity": metrics.get('avg_l1_sparsity', torch.tensor(0.0)).item() # Use the average L1 sparsity
                }
                # Add per-stream metrics
                for stream in model.streams:
                    log_data[f'mse_loss/{stream}'] = metrics.get(f'mse_loss_{stream}', torch.tensor(0.0)).item()
                    log_data[f'num_active_neurons/{stream}'] = metrics.get(f'num_active_neurons_{stream}', torch.tensor(0.0)).item() # Log per-stream active count
                    log_data[f'sparsity/{stream}'] = metrics.get(f'l1_sparsity_{stream}', torch.tensor(0.0)).item() # Log per-stream L1 sparsity
                    if model.auxk is not None:
                        log_data[f'auxk_loss/{stream}'] = metrics.get(f'auxk_loss_{stream}', torch.tensor(0.0)).item()
                # Add cross-loss metrics
                if cross_loss_coef > 0:
                    for src_stream in model.streams:
                        for tgt_stream in model.streams:
                            if src_stream == tgt_stream: continue
                            key = f'cross_loss_{tgt_stream}_from_{src_stream}'
                            log_data[f'cross_loss/{tgt_stream}_from_{src_stream}'] = metrics.get(key, torch.tensor(0.0)).item()
                    # Also log the averaged cross-loss if available in metrics
                    log_data['avg_cross_loss'] = metrics.get('avg_cross_loss', torch.tensor(0.0)).item()
                
                wandb_run.log(log_data)
            # -------------------------------------

            # Log individual stream metrics from the batch
            for stream in model.streams:
                recon_key = f'mse_loss_{stream}'
                active_key = f'num_active_neurons_{stream}'
                sparsity_key = f'l1_sparsity_{stream}'
                
                batch_recon_losses[stream].append(metrics.get(recon_key, torch.tensor(0.0)).item())
                batch_active_neurons[stream].append(metrics.get(active_key, torch.tensor(0.0)).item())
                batch_l1_sparsities[stream].append(metrics.get(sparsity_key, torch.tensor(0.0)).item())
                
                if model.auxk is not None:
                    auxk_key = f'auxk_loss_{stream}'
                    batch_auxk_losses[stream].append(metrics.get(auxk_key, torch.tensor(0.0)).item())

            # Log average cross loss if applicable
            if cross_loss_coef > 0:
                 # Log individual cross losses
                 for src_stream in model.streams:
                     for tgt_stream in model.streams:
                         if src_stream == tgt_stream: continue
                         key = f'cross_loss_{tgt_stream}_from_{src_stream}'
                         batch_cross_losses[key].append(metrics.get(key, torch.tensor(0.0)).item())

            # Update progress bar using averaged metrics
            epoch_loss_sum += loss.item()
            avg_sparsity = metrics.get('avg_sparsity', torch.tensor(0.0)).item()
            # Fetch the OVERALL average number of active neurons for the progress bar
            avg_num_active_batch = metrics.get('avg_num_active_neurons', torch.tensor(0.0)).item()
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                # Display avg_num_active_batch in the postfix
                "AvgAct": f"{avg_num_active_batch:.1f}/{model.k}", 
            })
            
            # Check for dead neurons (using per-stream stats)
            if global_step % 100 == 0:
                dead_neurons_dict = model.get_dead_neurons(dead_neuron_threshold)
                
                # Calculate per-stream dead counts 
                per_stream_dead_counts = {stream: mask.sum().item() for stream, mask in dead_neurons_dict.items()}
                
                # Check if any stream has dead neurons
                if any(count > 0 for count in per_stream_dead_counts.values()):                    
                    print(f"\nReinitializing dead neurons per stream at step {global_step}...")
                    print(f"  Dead counts: {per_stream_dead_counts}") # Print the counts
                    if wandb_run: # Log per-stream dead counts
                         log_per_stream_dead = {f"dead_neurons/{stream}": count for stream, count in per_stream_dead_counts.items()}
                         log_per_stream_dead["step"] = global_step
                         wandb_run.log(log_per_stream_dead)
                         
                    # Reinitialize dead neurons PER STREAM
                    with torch.no_grad():
                        for stream, stream_dead_mask in dead_neurons_dict.items():
                            # Check if this specific stream has dead neurons
                            if stream_dead_mask.any(): 
                                if stream in model.encoders and stream in model.latent_biases:
                                    enc = model.encoders[stream]
                                    lat_b = model.latent_biases[stream]
                                    
                                    # Reinitialize this stream's parameters using its specific mask
                                    enc.weight.data[stream_dead_mask] = torch.randn_like(enc.weight.data[stream_dead_mask]) * 0.01
                                    lat_b.data[stream_dead_mask] = torch.zeros_like(lat_b.data[stream_dead_mask])
                                    
                                    # Reset this stream's dead neuron counter using its specific mask
                                    try:
                                        stream_stats = getattr(model, f"stats_last_nonzero_{stream}")
                                        stream_stats[stream_dead_mask] = 0
                                    except AttributeError:
                                        print(f"Warning: Could not find stats_last_nonzero_{stream} buffer to reset.")
            
            global_step += 1
        
        # --- End of epoch --- 
        avg_epoch_total_loss = epoch_loss_sum / len(dataloader)
        epoch_total_losses.append(avg_epoch_total_loss)
        print(f"\nEpoch {epoch+1}/{num_epochs} - Avg Total Loss: {avg_epoch_total_loss:.6f}")
        
        # Append batch metrics to overall lists and calculate epoch averages per stream
        print("  --- Epoch Averages Per Stream ---")
        # Prepare epoch summary data for potential wandb logging
        epoch_log_data = {"epoch": epoch + 1, "avg_epoch_total_loss": avg_epoch_total_loss}
        # Iterate over streams for logging
        for stream in model.streams:
            # Calculate epoch averages from batch metrics
            # Note: batch_active_neurons now stores the *number* of active neurons per batch
            avg_recon = np.mean(batch_recon_losses[stream]) if batch_recon_losses[stream] else 0.0
            avg_active = np.mean(batch_active_neurons[stream]) if batch_active_neurons[stream] else 0.0
            # Calculate average L1 sparsity for the epoch
            avg_l1_sparsity = np.mean(batch_l1_sparsities[stream]) if batch_l1_sparsities[stream] else 0.0
            
            # Append epoch averages to the main lists
            recon_losses[stream].append(avg_recon)
            active_neurons[stream].append(avg_active)
            l1_sparsities[stream].append(avg_l1_sparsity) 
            
            # Add averages to wandb epoch log data
            epoch_log_data[f"avg_recon_loss/{stream}"] = avg_recon
            epoch_log_data[f"avg_num_active_neurons/{stream}"] = avg_active 
            epoch_log_data[f"avg_sparsity/{stream}"] = avg_l1_sparsity 

            print(f"    {stream.upper()}: Recon Loss: {avg_recon:.6f}, Avg Active Neurons: {avg_active:.2f}, Avg L1 Sparsity: {avg_l1_sparsity:.6f}", end="")
            if model.auxk is not None:
                # Calculate and append epoch average for auxk
                avg_auxk = np.mean(batch_auxk_losses[stream]) if batch_auxk_losses[stream] else 0.0
                auxk_losses[stream].append(avg_auxk)
                print(f", AuxK Loss: {avg_auxk:.6f}")
                epoch_log_data[f"avg_auxk_loss/{stream}"] = avg_auxk 
            else:
                 print()

        # Log average cross loss for the epoch
        if cross_loss_coef > 0:
            print("  --- Epoch Averages Cross-Loss --- ")
            total_epoch_cross_loss = 0
            count_epoch_cross_loss = 0
            # Calculate and append epoch average cross losses
            for key, batch_vals in batch_cross_losses.items():
                avg_epoch_cross = np.mean(batch_vals) if batch_vals else 0.0
                # Append epoch average to the main list
                cross_losses[key].append(avg_epoch_cross)
                print(f"    Avg {key}: {avg_epoch_cross:.6f}")
                epoch_log_data[f"avg_{key}"] = avg_epoch_cross 
                total_epoch_cross_loss += avg_epoch_cross
                count_epoch_cross_loss += 1
            # Calculate and log overall average cross loss for the epoch
            if count_epoch_cross_loss > 0:
                 epoch_log_data["avg_epoch_cross_loss"] = total_epoch_cross_loss / count_epoch_cross_loss
        
        # Run validation if validation dataloader is provided
        if val_dataloader is not None:
            model.eval()  # Set model to evaluation mode
            val_loss_sum = 0.0
            val_batch_recon_losses = defaultdict(list)
            val_batch_active_neurons = defaultdict(list)
            val_batch_l1_sparsities = defaultdict(list)
            val_batch_auxk_losses = defaultdict(list)
            val_batch_cross_losses = defaultdict(list)
            
            print("\n  --- Running Validation ---")
            with torch.no_grad():
                for val_batch_idx, val_batch_data in enumerate(val_dataloader):
                    # Process validation data similar to training
                    if not isinstance(val_batch_data, dict):
                        raise ValueError(f"Val dataloader expected to yield dict, but got {type(val_batch_data)}")
                    
                    val_inputs = {}
                    for stream_name, data_np in val_batch_data.items():
                        if not isinstance(data_np, np.ndarray):
                            try:
                                data_np = np.array(data_np)
                            except Exception as e:
                                raise TypeError(f"Could not convert validation batch data for stream '{stream_name}' to NumPy array: {e}")
                        
                        val_inputs[stream_name] = torch.from_numpy(data_np).to(device, non_blocking=True)
                    
                    if not val_inputs:
                        print(f"Warning: Skipping validation batch {val_batch_idx} due to no matching streams found in data.")
                        continue
                    
                    # Forward pass and compute loss
                    val_output = model(val_inputs)
                    val_loss, val_metrics = model.compute_loss(val_output, val_inputs, auxk_coef, cross_loss_coef)
                    val_loss_sum += val_loss.item()
                    
                    # Collect validation metrics
                    for stream in model.streams:
                        recon_key = f'mse_loss_{stream}'
                        active_key = f'num_active_neurons_{stream}'
                        sparsity_key = f'l1_sparsity_{stream}'
                        
                        val_batch_recon_losses[stream].append(val_metrics.get(recon_key, torch.tensor(0.0)).item())
                        val_batch_active_neurons[stream].append(val_metrics.get(active_key, torch.tensor(0.0)).item())
                        val_batch_l1_sparsities[stream].append(val_metrics.get(sparsity_key, torch.tensor(0.0)).item())
                        
                        if model.auxk is not None:
                            auxk_key = f'auxk_loss_{stream}'
                            val_batch_auxk_losses[stream].append(val_metrics.get(auxk_key, torch.tensor(0.0)).item())
                    
                    # Collect validation cross-loss if applicable
                    if cross_loss_coef > 0:
                        for src_stream in model.streams:
                            for tgt_stream in model.streams:
                                if src_stream == tgt_stream: continue
                                key = f'cross_loss_{tgt_stream}_from_{src_stream}'
                                val_batch_cross_losses[key].append(val_metrics.get(key, torch.tensor(0.0)).item())
            
            # Calculate validation epoch averages
            val_avg_epoch_total_loss = val_loss_sum / len(val_dataloader)
            val_epoch_total_losses.append(val_avg_epoch_total_loss)
            
            # Prepare validation log data for wandb
            val_log_data = {"epoch": epoch + 1, "val_avg_epoch_total_loss": val_avg_epoch_total_loss}
            
            print(f"  Validation - Avg Total Loss: {val_avg_epoch_total_loss:.6f}")
            print("  --- Validation Averages Per Stream ---")
            
            # Calculate and log validation metrics per stream
            for stream in model.streams:
                val_avg_recon = np.mean(val_batch_recon_losses[stream]) if val_batch_recon_losses[stream] else 0.0
                val_avg_active = np.mean(val_batch_active_neurons[stream]) if val_batch_active_neurons[stream] else 0.0
                val_avg_l1_sparsity = np.mean(val_batch_l1_sparsities[stream]) if val_batch_l1_sparsities[stream] else 0.0
                
                # Append validation averages to lists
                val_recon_losses[stream].append(val_avg_recon)
                val_active_neurons[stream].append(val_avg_active)
                val_l1_sparsities[stream].append(val_avg_l1_sparsity)
                
                # Add validation averages to wandb log data
                val_log_data[f"val_avg_recon_loss/{stream}"] = val_avg_recon
                val_log_data[f"val_avg_num_active_neurons/{stream}"] = val_avg_active
                val_log_data[f"val_avg_sparsity/{stream}"] = val_avg_l1_sparsity
                
                print(f"    {stream.upper()}: Recon Loss: {val_avg_recon:.6f}, Avg Active Neurons: {val_avg_active:.2f}, Avg L1 Sparsity: {val_avg_l1_sparsity:.6f}", end="")
                
                if model.auxk is not None:
                    val_avg_auxk = np.mean(val_batch_auxk_losses[stream]) if val_batch_auxk_losses[stream] else 0.0
                    val_auxk_losses[stream].append(val_avg_auxk)
                    print(f", AuxK Loss: {val_avg_auxk:.6f}")
                    val_log_data[f"val_avg_auxk_loss/{stream}"] = val_avg_auxk
                else:
                    print()
            
            # Log validation cross-loss if applicable
            if cross_loss_coef > 0:
                print("  --- Validation Averages Cross-Loss ---")
                val_total_epoch_cross_loss = 0
                val_count_epoch_cross_loss = 0
                
                for key, batch_vals in val_batch_cross_losses.items():
                    val_avg_epoch_cross = np.mean(batch_vals) if batch_vals else 0.0
                    val_cross_losses[key].append(val_avg_epoch_cross)
                    print(f"    Avg {key}: {val_avg_epoch_cross:.6f}")
                    val_log_data[f"val_avg_{key}"] = val_avg_epoch_cross
                    val_total_epoch_cross_loss += val_avg_epoch_cross
                    val_count_epoch_cross_loss += 1
                
                if val_count_epoch_cross_loss > 0:
                    val_log_data["val_avg_epoch_cross_loss"] = val_total_epoch_cross_loss / val_count_epoch_cross_loss
            
            # Log validation metrics to wandb
            if wandb_run:
                wandb_run.log(val_log_data)
            
            # Return to training mode
            model.train()
        
        # Log epoch summary to Wandb (if enabled)
        if wandb_run:
            wandb_run.log(epoch_log_data)

        # Calculate and log dead neurons at the end of each epoch
        dead_neurons_dict = model.get_dead_neurons(dead_neuron_threshold)
        per_stream_dead_counts = {stream: mask.sum().item() for stream, mask in dead_neurons_dict.items()}
        
        print("\n  --- End of Epoch Dead Neuron Counts ---")
        for stream, count in per_stream_dead_counts.items():
            print(f"    {stream.upper()}: {count} dead neurons")
            epoch_dead_neurons[stream].append(count)
            
            # Add to wandb logging
            if wandb_run:
                wandb_run.log({
                    "epoch": epoch + 1,
                    f"epoch_dead_neurons/{stream}": count
                })

    print("\nTraining the MS SAE finished.")
    
    # Return metrics containing lists of epoch averages
    return_metrics = {
        "recon_losses": dict(recon_losses), 
        "active_neurons": dict(active_neurons), 
        "l1_sparsities": dict(l1_sparsities), 
        "epoch_dead_neurons": dict(epoch_dead_neurons),
    }
    if model.auxk is not None:
        return_metrics["auxk_losses"] = dict(auxk_losses)
    if cross_loss_coef > 0:
        return_metrics["cross_losses"] = dict(cross_losses)
    
    # Add validation metrics to return dictionary
    if val_dataloader is not None:
        return_metrics["val_epoch_total_losses"] = val_epoch_total_losses
        return_metrics["val_recon_losses"] = dict(val_recon_losses)
        return_metrics["val_active_neurons"] = dict(val_active_neurons)
        return_metrics["val_l1_sparsities"] = dict(val_l1_sparsities)
        if model.auxk is not None:
            return_metrics["val_auxk_losses"] = dict(val_auxk_losses)
        if cross_loss_coef > 0:
            return_metrics["val_cross_losses"] = dict(val_cross_losses)
        
    return model, return_metrics
