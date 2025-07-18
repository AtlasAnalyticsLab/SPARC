import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from sparc.datasets import HDF5FeatureDataset
from typing import Dict
import random
import numpy as np
import os

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # These may slow training, but they are used for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def return_top_activating_images(
    stream_feature_files: Dict[str, str],
    model,
    batch_size: int = 64,
    seed: int = 42,
    verbose: bool = True
) -> dict:
    """
    Uses a trained MultiStreamSparseAutoencoder model to compute top activating samples,
    raw inputs, logits, latents (sparse codes), self-reconstructions,
    top-k indices (per stream), and cross-reconstructions for EACH stream.

    Args:
        stream_feature_files (Dict[str, str]): Dictionary mapping stream names to HDF5 file paths.
        model (MultiStreamSparseAutoencoder): An initialized and potentially loaded MultiStreamSparseAutoencoder model.
        batch_size (int): Batch size. Defaults to 64.
        seed (int): Random seed. Defaults to 42.
        verbose (bool): If True, print status messages. Defaults to True.

    Returns:
        dict: A dictionary containing:
            - 'dataset_indices': List of original dataset indices batches (numpy arrays).
            - For each stream (e.g., 'dino'): Dictionary with:
                - 'top_activations': List (per latent) of sorted (activation, index) tuples.
                - 'raw': List of raw input batches (numpy arrays).
                - 'logits': List of encoder logit batches (numpy arrays).
                - 'latents': List of sparse code batches (post-activation) (numpy arrays).
                - 'recon': List of self-reconstruction batches (numpy arrays).
                - 'topk_indices': List of TopK indices batches for this stream (numpy arrays).
            - 'cross_reconstructions': Dict containing lists of batch numpy arrays for each
              cross-stream reconstruction (e.g., 'clip_from_dino', 'text_from_dino', etc.).
    """
    set_seed(seed)
    split_generator = torch.Generator()
    split_generator.manual_seed(seed)
    
    if verbose:
        print(f"Loading features for analysis from {len(stream_feature_files)} files:")
        for stream_name, file_path in stream_feature_files.items():
            print(f"  {stream_name.upper()}: {file_path}")
            
    dataset = HDF5FeatureDataset(
        stream_files=stream_feature_files,
        return_index=True
    )
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, 
                            worker_init_fn=seed_worker, generator=split_generator)

    streams = dataset.streams
    d_streams = dataset.get_feature_dims()
    n_latents = model.n_latents
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    if verbose: print(f"Computing activations and reconstructions using provided model for streams: {streams}...")
    
    results = {stream: {'top_activations': [[] for _ in range(n_latents)],
                     'raw': [],
                     'logits': [],
                     'latents': [],
                     'recon': [],
                     'topk_indices': []}
               for stream in streams}
    results['cross_reconstructions'] = defaultdict(list)
    all_dataset_indices = []

    with torch.no_grad():
        for batch_data, dataset_indices_batch in tqdm(val_loader):
            all_dataset_indices.append(dataset_indices_batch.numpy())

            inputs = {}
            for stream_name, data_tensor in batch_data.items():
                if stream_name in streams:
                    inputs[stream_name] = data_tensor.to(device, non_blocking=True)
                else:
                     if verbose: print(f"Warning: Skipping unexpected stream '{stream_name}' from dataloader during analysis.")
            
            if not inputs:
                if verbose: print("Warning: Skipping batch due to no matching streams found in data.")
                continue
                
            outputs = model(inputs)
            
            shared_indices_batch = None
            if 'shared_indices' in outputs:
                shared_indices_batch = outputs['shared_indices'].cpu().numpy()
            
            for stream_name, input_tensor in inputs.items():
                results[stream_name]['raw'].append(input_tensor.cpu().numpy())
                logits_key = f'logits_{stream_name}'
                if logits_key in outputs:
                    results[stream_name]['logits'].append(outputs[logits_key].cpu().numpy())
            
            for stream in model.streams: 
                sparse_codes_key = f'sparse_codes_{stream}'
                recon_key = f'recon_{stream}'
                
                # Use shared indices if available, otherwise use per-stream
                if shared_indices_batch is not None:
                    results[stream]['topk_indices'].append(shared_indices_batch)
                else:
                    indices_key = f'indices_{stream}'
                    if indices_key in outputs:
                        results[stream]['topk_indices'].append(outputs[indices_key].cpu().numpy())
                
                latent_activations = outputs[sparse_codes_key]
                reconstructions = outputs[recon_key]

                results[stream]['latents'].append(latent_activations.cpu().numpy())
                results[stream]['recon'].append(reconstructions.cpu().numpy())
                
                # Calculate top activations for this stream
                for dim in range(n_latents):
                    activations = latent_activations[:, dim].cpu().numpy()
                    for i, activation in enumerate(activations):
                        results[stream]['top_activations'][dim].append((activation, dataset_indices_batch[i].item()))

            for src_stream in model.streams:
                for tgt_stream in model.streams:
                    if src_stream == tgt_stream: continue
                    cross_key = f'cross_recon_{tgt_stream}_from_{src_stream}'
                    if cross_key in outputs:
                        cross_recon_tensor = outputs[cross_key]
                        results['cross_reconstructions'][f'{tgt_stream}_from_{src_stream}'].append(cross_recon_tensor.cpu().numpy())

    # Sort activations for each stream and dimension
    if verbose: print(f"Sorting activations for {streams}...")
    for stream in streams:
        for dim in range(n_latents):
            results[stream]['top_activations'][dim].sort(key=lambda x: x[0], reverse=True)
    
    dataset.close()
    results['dataset_indices'] = all_dataset_indices 

    print("Latent Analysis on Top Activating Images Completed!")
    return results