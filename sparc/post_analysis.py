import h5py
import numpy as np
import json
from typing import Dict, List, Any, Union, Optional, Tuple
import os
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import defaultdict
from scipy.sparse import csr_matrix

from sparc.utils import HDF5FeatureDataset, seed_worker, set_seed

def _load_h5_group_as_list_of_arrays(h5_group: h5py.Group) -> List[np.ndarray]:
    """Loads datasets named batch_0, batch_1, ... from an HDF5 group into a list of numpy arrays."""
    loaded_list = []
    # Ensure keys are sorted numerically by batch index
    batch_keys = sorted([key for key in h5_group.keys() if key.startswith('batch_')], 
                        key=lambda k: int(k.split('_')[1]))
    for key in batch_keys:
        loaded_list.append(h5_group[key][:])
    return loaded_list

def _load_top_activations_from_h5_group(top_act_h5_group: h5py.Group) -> List[List[tuple[float, int]]]:
    """Loads the top_activations structure from an HDF5 group."""
    num_latents = 0
    latent_keys = [k for k in top_act_h5_group.keys() if k.startswith('latent_')]
    if latent_keys:
        # Determine num_latents by finding the max index from latent_X keys
        num_latents = max(int(k.split('_')[1]) for k in latent_keys) + 1 
    
    reconstructed_list = [[] for _ in range(num_latents)]
    for i in range(num_latents):
        latent_key = f'latent_{i}'
        if latent_key in top_act_h5_group:
            latent_group = top_act_h5_group[latent_key]
            if 'activation_values' in latent_group and 'original_sample_indices' in latent_group:
                activations = latent_group['activation_values'][:]
                indices = latent_group['original_sample_indices'][:]
                # Ensure activations and indices are not empty before zipping
                if activations.ndim > 0 and indices.ndim > 0 and len(activations) > 0:
                    reconstructed_list[i] = list(zip(activations, indices))
                else:
                    reconstructed_list[i] = []
            else:
                reconstructed_list[i] = []
        # If latent_key is missing, it means no activations were saved for it (e.g. if n_latents was too high)
        # The list will correctly have an empty list at this position due to pre-initialization.

    return reconstructed_list

def load_analysis_from_h5(filepath: str) -> Dict[str, Any]:
    """
    Loads the analysis results dictionary from an HDF5 file,
    which was saved by _compute_and_save_analysis_incrementally_to_h5.
    """
    results: Dict[str, Any] = {}
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"HDF5 cache file not found: {filepath}")

    with h5py.File(filepath, 'r') as hf:
        if 'dataset_indices' in hf:
            results['dataset_indices'] = _load_h5_group_as_list_of_arrays(hf['dataset_indices'])
        else:
            results['dataset_indices'] = []

        if 'shared_topk_indices' in hf:
            results['shared_topk_indices'] = _load_h5_group_as_list_of_arrays(hf['shared_topk_indices'])
            
        # Identify stream groups
        # These are top-level groups that are not 'dataset_indices', 'cross_reconstructions', or 'shared_topk_indices'
        stream_names = [k for k in hf.keys() if k not in ['dataset_indices', 'cross_reconstructions', 'shared_topk_indices']]
        
        for stream_name in stream_names:
            if stream_name not in hf or not isinstance(hf[stream_name], h5py.Group):
                continue
            stream_group = hf[stream_name]
            results[stream_name] = {}
            
            for key in stream_group.keys():
                item = stream_group[key]
                if isinstance(item, h5py.Group):
                    if key == 'top_activations':
                        results[stream_name][key] = _load_top_activations_from_h5_group(item)
                    else:
                        results[stream_name][key] = _load_h5_group_as_list_of_arrays(item)

        results['cross_reconstructions'] = {}
        if 'cross_reconstructions' in hf and isinstance(hf['cross_reconstructions'], h5py.Group):
            cross_recon_group = hf['cross_reconstructions']
            for cross_key in cross_recon_group.keys():
                if isinstance(cross_recon_group[cross_key], h5py.Group):
                     results['cross_reconstructions'][cross_key] = _load_h5_group_as_list_of_arrays(cross_recon_group[cross_key])
                
    return results

def _ensure_h5_group(hf: h5py.File, path: str) -> h5py.Group:
    """Ensures an HDF5 group exists, creating it if necessary."""
    if path not in hf:
        return hf.create_group(path)
    return hf[path]

def _compute_and_save_analysis_incrementally_to_h5(
    stream_feature_files: Dict[str, str],
    model: torch.nn.Module,
    output_h5_filepath: str,
    batch_size: int = 64,
    seed: int = 42,
    verbose: bool = True
):
    """
    Computes analysis results and saves them incrementally to an HDF5 file.
    This is a memory-efficient alternative to return_top_activating_images.
    """
    set_seed(seed)
    split_generator = torch.Generator()
    split_generator.manual_seed(seed)

    if verbose:
        print(f"Loading features for incremental HDF5 analysis from {len(stream_feature_files)} files:")
        for stream_name, file_path in stream_feature_files.items():
            print(f"  {stream_name.upper()}: {file_path}")

    dataset = HDF5FeatureDataset(stream_files=stream_feature_files, return_index=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                            worker_init_fn=seed_worker, generator=split_generator)

    streams_from_dataset = dataset.streams
    n_latents = model.n_latents
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Structure: {stream_name: [ [ (act, idx), ... ], ... for each_latent ] }
    # Use a fixed-size buffer per latent to reduce memory consumption
    max_activations_to_store = 200
    top_activations_buffer = {stream: [[] for _ in range(n_latents)] for stream in streams_from_dataset}

    with h5py.File(output_h5_filepath, 'w') as hf:
        dsi_group = _ensure_h5_group(hf, 'dataset_indices')
        shared_idx_group = None
        
        stream_h5_groups = {}
        for stream_name in streams_from_dataset:
            s_group = _ensure_h5_group(hf, stream_name)
            stream_h5_groups[stream_name] = {
                'raw': _ensure_h5_group(s_group, 'raw'),
                'recon': _ensure_h5_group(s_group, 'recon'),
                'topk_indices': _ensure_h5_group(s_group, 'topk_indices')
            }

        cr_base_group_for_file = None

        if verbose: 
            print(f"Computing and saving activations incrementally to {output_h5_filepath}...")
            # Create a progress bar with more information
            total_batches = len(val_loader)
            progress_bar = tqdm(
                total=total_batches,
                desc="Processing batches",
                unit="batch",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                disable=not verbose
            )
            total_samples_processed = 0

        for batch_idx, (batch_data, dataset_indices_batch) in enumerate(val_loader):
            batch_size_actual = len(dataset_indices_batch)
            
            dsi_group.create_dataset(f'batch_{batch_idx}', data=dataset_indices_batch.numpy())

            inputs = {}
            for stream_name, data_tensor in batch_data.items():
                if stream_name in streams_from_dataset:
                    inputs[stream_name] = data_tensor.to(device, non_blocking=True)
            
            if not inputs: 
                if verbose:
                    progress_bar.update(1)
                continue

            with torch.no_grad():
                outputs = model(inputs)

            shared_indices_batch_np = None
            if 'shared_indices' in outputs:
                if shared_idx_group is None:
                    shared_idx_group = _ensure_h5_group(hf, 'shared_topk_indices')
                shared_indices_batch_np = outputs['shared_indices'].cpu().numpy()
                if shared_idx_group is not None:
                    shared_idx_group.create_dataset(f'batch_{batch_idx}', data=shared_indices_batch_np)

            for stream_name, input_tensor in inputs.items():
                stream_h5_groups[stream_name]['raw'].create_dataset(f'batch_{batch_idx}', data=input_tensor.cpu().numpy())
            
            for stream in model.streams:
                if stream not in streams_from_dataset: continue

                sparse_codes_key = f'sparse_codes_{stream}'
                recon_key = f'recon_{stream}'
                
                if sparse_codes_key in outputs:
                    latent_activations = outputs[sparse_codes_key]
                    # Instead of saving full sparse tensor, save only non-zero values and indices
                    stream_main_h5_group = hf[stream]
                    latents_group = _ensure_h5_group(stream_main_h5_group, 'latents_sparse')
                    
                    non_zero_mask = latent_activations != 0
                    values = latent_activations[non_zero_mask]
                    
                    batch_indices, latent_indices = torch.where(non_zero_mask)
                    
                    batch_group = _ensure_h5_group(latents_group, f'batch_{batch_idx}')
                    batch_group.create_dataset('values', data=values.cpu().numpy())
                    batch_group.create_dataset('batch_indices', data=batch_indices.cpu().numpy())
                    batch_group.create_dataset('latent_indices', data=latent_indices.cpu().numpy())
                    
                    batch_group.attrs['original_shape'] = latent_activations.shape
                    
                    # Update top_activations_buffer (keeping only top activations)
                    for dim in range(n_latents):
                        activations_for_dim = latent_activations[:, dim].cpu().numpy()
                        original_indices_for_batch = dataset_indices_batch.numpy()
                        
                        batch_activations = [(act, original_indices_for_batch[i].item()) 
                                            for i, act in enumerate(activations_for_dim) if act > 0]
                        
                        # Merge with existing buffer and keep only top activations
                        current_buffer = top_activations_buffer[stream][dim]
                        merged = current_buffer + batch_activations
                        # Sort by activation value (largest first) and keep only max_activations_to_store
                        merged.sort(key=lambda x: x[0], reverse=True)
                        top_activations_buffer[stream][dim] = merged[:max_activations_to_store]
                
                if recon_key in outputs:
                    stream_h5_groups[stream]['recon'].create_dataset(f'batch_{batch_idx}', data=outputs[recon_key].cpu().numpy())

                if shared_indices_batch_np is None:
                    indices_key = f'indices_{stream}'
                    if indices_key in outputs:
                        stream_h5_groups[stream]['topk_indices'].create_dataset(f'batch_{batch_idx}', data=outputs[indices_key].cpu().numpy())
            
            has_cross_recons = False
            for src_stream_cr in model.streams:
                for tgt_stream_cr in model.streams:
                    if src_stream_cr == tgt_stream_cr: continue
                    if f'cross_recon_{tgt_stream_cr}_from_{src_stream_cr}' in outputs:
                        has_cross_recons = True
                        break
                if has_cross_recons: break
            
            if has_cross_recons and cr_base_group_for_file is None:
                cr_base_group_for_file = _ensure_h5_group(hf, 'cross_reconstructions')

            if cr_base_group_for_file is not None:
                for src_stream in model.streams:
                    for tgt_stream in model.streams:
                        if src_stream == tgt_stream: continue
                        cross_key = f'cross_recon_{tgt_stream}_from_{src_stream}'
                        if cross_key in outputs:
                            cross_recon_tensor = outputs[cross_key]
                            specific_cross_group = _ensure_h5_group(cr_base_group_for_file, f'{tgt_stream}_from_{src_stream}')
                            specific_cross_group.create_dataset(f'batch_{batch_idx}', data=cross_recon_tensor.cpu().numpy())

            if verbose:
                total_samples_processed += batch_size_actual
                progress_bar.set_postfix({
                    'samples': total_samples_processed, 
                    'batch_size': batch_size_actual,
                    'streams': len(streams_from_dataset)
                })
                progress_bar.update(1)

        if verbose:
            progress_bar.close()
            print(f"Sorting and saving top activations to {output_h5_filepath}...")
            # Create another progress bar for the sorting phase
            sorting_progress = tqdm(
                total=len(streams_from_dataset) * n_latents,
                desc="Sorting top activations",
                unit="latent",
                disable=not verbose
            )

        for stream_name_ta in streams_from_dataset:
            if stream_name_ta not in hf: continue 
            stream_main_h5_group = hf[stream_name_ta]
            top_act_h5_group = _ensure_h5_group(stream_main_h5_group, 'top_activations')
            for dim_idx in range(n_latents):
                # Buffer is already sorted during the incremental updates
                sorted_activations = top_activations_buffer[stream_name_ta][dim_idx]
                activations_to_save = [(act, idx) for act, idx in sorted_activations if act > 0.0]

                latent_dim_h5_group = _ensure_h5_group(top_act_h5_group, f'latent_{dim_idx}')
                
                if activations_to_save:
                    act_values = np.array([item[0] for item in activations_to_save], dtype=np.float32)
                    orig_indices = np.array([item[1] for item in activations_to_save], dtype=np.int64)
                    latent_dim_h5_group.create_dataset('activation_values', data=act_values)
                    latent_dim_h5_group.create_dataset('original_sample_indices', data=orig_indices)
                else:
                    latent_dim_h5_group.create_dataset('activation_values', data=np.array([], dtype=np.float32))
                    latent_dim_h5_group.create_dataset('original_sample_indices', data=np.array([], dtype=np.int64))
                
                if verbose and sorting_progress:
                    sorting_progress.update(1)
        
        if verbose:
            sorting_progress.close()

    dataset.close()
    if verbose: print(f"Incremental analysis and saving to {output_h5_filepath} complete.")


class HDF5AnalysisResultsDataset(Dataset):
    """
    A PyTorch Dataset for lazily loading analysis results from an HDF5 file
    generated by _compute_and_save_analysis_incrementally_to_h5.

    This dataset allows access to per-sample analysis data without loading the
    entire results file into memory.
    """
    def __init__(self, analysis_h5_filepath: str, batch_size_used_for_generation: int, verbose_load: bool = False):
        """
        Args:
            analysis_h5_filepath: Path to the HDF5 analysis results file.
            batch_size_used_for_generation: The batch size used when the HDF5 file was created.
            verbose_load: If True, prints warnings during data loading attempts by accessor methods.
        """
        if not os.path.exists(analysis_h5_filepath):
            raise FileNotFoundError(f"Analysis HDF5 file not found: {analysis_h5_filepath}")

        self.filepath = analysis_h5_filepath
        self.batch_size = batch_size_used_for_generation
        self.verbose_load = verbose_load
        self.hf: h5py.File = h5py.File(self.filepath, 'r')

        self.streams: List[str] = []
        self._n_latents: int = 0
        self.num_samples: int = 0
        
        self.batch_lengths: List[int] = []
        self.batch_offsets: List[int] = []
        self._original_dataset_indices_all: Optional[np.ndarray] = None

        self._initialize_metadata()

    def _initialize_metadata(self):
        """Reads metadata from the HDF5 file to set up the dataset."""
        if 'dataset_indices' not in self.hf or not isinstance(self.hf['dataset_indices'], h5py.Group):
            self.close()
            raise ValueError("HDF5 file missing 'dataset_indices' group or it's not a group.")

        current_offset = 0
        dsi_group = self.hf['dataset_indices']
        batch_keys = sorted([key for key in dsi_group.keys() if key.startswith('batch_')],
                            key=lambda k: int(k.split('_')[1]))
        if not batch_keys:
            self.close()
            raise ValueError("No 'batch_X' datasets found in 'dataset_indices' group.")
            
        for key in batch_keys:
            batch_len = dsi_group[key].shape[0]
            self.batch_lengths.append(batch_len)
            self.batch_offsets.append(current_offset)
            current_offset += batch_len
        self.num_samples = current_offset

        # Determine streams (top-level groups excluding reserved ones)
        reserved_keys = ['dataset_indices', 'cross_reconstructions', 'shared_topk_indices']
        self.streams = [k for k in self.hf.keys() if k not in reserved_keys and isinstance(self.hf[k], h5py.Group)]

        if not self.streams:
            # No stream data, but could still be useful if only dataset_indices are needed.
            print("Warning: No stream-specific data groups found in the HDF5 file.")

        # Determine n_latents from the first available stream's top_activations or latents group
        if self.streams:
            example_stream_name = self.streams[0]
            if example_stream_name in self.hf and 'top_activations' in self.hf[example_stream_name]:
                top_act_group = self.hf[example_stream_name]['top_activations']
                latent_keys = [k for k in top_act_group.keys() if k.startswith('latent_')]
                if latent_keys:
                    self._n_latents = max(int(k.split('_')[1]) for k in latent_keys) + 1
            elif example_stream_name in self.hf and 'latents' in self.hf[example_stream_name]:
                # Fallback: check a latents batch dataset shape if top_activations is not structured as expected
                latents_group = self.hf[example_stream_name]['latents']
                if 'batch_0' in latents_group:
                    self._n_latents = latents_group['batch_0'].shape[1]
        
        if self._n_latents == 0 and self.streams:
             print(f"Warning: Could not determine n_latents from stream '{self.streams[0]}'. It will be 0.")

    def __len__(self) -> int:
        return self.num_samples

    def _find_batch_info(self, global_sample_idx: int) -> Tuple[int, int]:
        """Given a global sample index, find the batch number and index within that batch."""
        if not (0 <= global_sample_idx < self.num_samples):
            raise IndexError(f"Global sample index {global_sample_idx} out of range (0-{self.num_samples-1}).")
        
        # Find the batch this global index falls into
        batch_idx = -1
        for i, offset in enumerate(self.batch_offsets):
            if global_sample_idx < offset + self.batch_lengths[i]:
                batch_idx = i
                break
        
        local_idx_in_batch = global_sample_idx - self.batch_offsets[batch_idx]
        return batch_idx, local_idx_in_batch

    def _get_data_from_batched_group(self, group_path_str: str, batch_idx: int, local_idx_in_batch: int) -> Optional[np.ndarray]:
        """Helper to fetch a single item from a dataset within a batched group."""
        if group_path_str not in self.hf:
            return None
        group = self.hf[group_path_str]
        batch_dataset_name = f'batch_{batch_idx}'
        if batch_dataset_name not in group:
            return None
        return group[batch_dataset_name][local_idx_in_batch]

    def _get_sparse_latents(self, stream_name: str, batch_idx: int, local_idx_in_batch: int) -> Optional[np.ndarray]:
        """Helper to reconstruct sparse latents from values and indices."""
        if stream_name not in self.hf or not isinstance(self.hf[stream_name], h5py.Group):
            return None
            
        stream_group = self.hf[stream_name]
        if 'latents_sparse' not in stream_group or not isinstance(stream_group['latents_sparse'], h5py.Group):
            return None
            
        latents_group = stream_group['latents_sparse']
        batch_group_name = f'batch_{batch_idx}'
        
        if batch_group_name not in latents_group or not isinstance(latents_group[batch_group_name], h5py.Group):
            return None
            
        batch_group = latents_group[batch_group_name]
        
        required_datasets = ['values', 'batch_indices', 'latent_indices']
        if not all(ds in batch_group for ds in required_datasets):
            return None
            
        if 'original_shape' not in batch_group.attrs:
            return None
            
        original_shape = batch_group.attrs['original_shape']
        
        values = batch_group['values'][:]
        batch_indices = batch_group['batch_indices'][:]
        latent_indices = batch_group['latent_indices'][:]
        
        # Find the indices that correspond to the requested sample
        sample_mask = batch_indices == local_idx_in_batch
        if not np.any(sample_mask):
            # No non-zero values for this sample, return zeros
            return np.zeros(original_shape[1], dtype=np.float32)
            
        sample_values = values[sample_mask]
        sample_latent_indices = latent_indices[sample_mask]
        
        reconstructed = np.zeros(original_shape[1], dtype=np.float32)
        reconstructed[sample_latent_indices] = sample_values
        
        return reconstructed

    def __getitem__(self, global_sample_idx: int) -> Dict[str, Any]:
        """
        Retrieves all analysis data for a single sample, specified by its global index
        across all processed samples (0 to N-1).
        """
        batch_idx, local_idx_in_batch = self._find_batch_info(global_sample_idx)

        sample_data: Dict[str, Any] = {}

        original_id_val = self._get_data_from_batched_group('dataset_indices', batch_idx, local_idx_in_batch)
        sample_data['original_dataset_id'] = original_id_val.item() if original_id_val is not None else None

        if 'shared_topk_indices' in self.hf:
            sample_data['shared_topk_indices'] = self._get_data_from_batched_group(
                'shared_topk_indices', batch_idx, local_idx_in_batch
            )

        for stream_name in self.streams:
            stream_data: Dict[str, Optional[np.ndarray]] = {}
            stream_group_path = stream_name

            data_types_to_fetch = ['raw', 'recon']
            
            for data_type in data_types_to_fetch:
                data_group_path = f'{stream_group_path}/{data_type}'
                stream_data[data_type] = self._get_data_from_batched_group(
                    data_group_path, batch_idx, local_idx_in_batch
                )
            
            stream_data['latents'] = self._get_sparse_latents(stream_name, batch_idx, local_idx_in_batch)
            
            # Per-stream topk_indices are only relevant if shared_topk_indices are not present
            if 'shared_topk_indices' not in self.hf:
                data_group_path = f'{stream_group_path}/topk_indices'
                stream_data['topk_indices'] = self._get_data_from_batched_group(
                    data_group_path, batch_idx, local_idx_in_batch
                )
                
            sample_data[stream_name] = stream_data
        
        sample_data['cross_reconstructions'] = {}
        if 'cross_reconstructions' in self.hf:
            cr_base_group = self.hf['cross_reconstructions']
            for cr_key in cr_base_group.keys():
                if isinstance(cr_base_group[cr_key], h5py.Group):
                    sample_data['cross_reconstructions'][cr_key] = self._get_data_from_batched_group(
                        f'cross_reconstructions/{cr_key}', batch_idx, local_idx_in_batch
                    )
        return sample_data

    def get_top_activations_for_latent(self, stream_name: str, latent_idx: int) -> List[Tuple[float, int]]:
        """Loads and returns sorted (activation, original_sample_idx) for a specific latent."""
        if stream_name not in self.streams:
            raise ValueError(f"Stream '{stream_name}' not found in dataset.")
        if not (0 <= latent_idx < self._n_latents):
            raise IndexError(f"Latent index {latent_idx} out of range (0-{self._n_latents-1}).")

        stream_h5_group = self.hf[stream_name]
        if 'top_activations' not in stream_h5_group:
            return []
        
        top_act_h5_group = stream_h5_group['top_activations']
        latent_key_in_h5 = f'latent_{latent_idx}'
        
        if latent_key_in_h5 not in top_act_h5_group:
            return []
            
        latent_group = top_act_h5_group[latent_key_in_h5]
        if 'activation_values' in latent_group and 'original_sample_indices' in latent_group:
            activations = latent_group['activation_values'][:]
            indices = latent_group['original_sample_indices'][:]
            if activations.ndim > 0 and indices.ndim > 0 and len(activations) > 0:
                return list(zip(activations, indices))
        return []

    def get_all_original_dataset_indices(self) -> np.ndarray:
        """Returns a concatenated numpy array of all original dataset indices."""
        if self._original_dataset_indices_all is not None:
            return self._original_dataset_indices_all
        
        all_indices = []
        if 'dataset_indices' not in self.hf:
            self._original_dataset_indices_all = np.array([], dtype=np.int64)
            return self._original_dataset_indices_all
            
        dsi_group = self.hf['dataset_indices']
        batch_keys = sorted([key for key in dsi_group.keys() if key.startswith('batch_')],
                            key=lambda k: int(k.split('_')[1]))
        for key in batch_keys:
            all_indices.append(dsi_group[key][:])
        
        if not all_indices:
            self._original_dataset_indices_all = np.array([], dtype=np.int64)
        else:
            self._original_dataset_indices_all = np.concatenate(all_indices)
        return self._original_dataset_indices_all

    def get_all_features_for_stream(self, stream_name: str, feature_type: str, return_sparse: bool = False) -> Optional[Union[np.ndarray, csr_matrix]]:
        """
        Retrieves all features for a given stream and feature type, concatenated into a single NumPy array
        or returned as a sparse matrix if feature_type is 'latents' and return_sparse is True.
        Feature types can be 'raw', 'logits', 'latents', 'recon', 'topk_indices'.
        Returns None if the stream or feature type is not found or has no data.
        """
        if stream_name not in self.hf or not isinstance(self.hf[stream_name], h5py.Group):
            if self.verbose_load: print(f"Warning: Stream '{stream_name}' not found in HDF5 file.")
            return None
        
        stream_group = self.hf[stream_name]
        
        # Special handling for latents which are stored in sparse format
        if feature_type == 'latents':
            if 'latents_sparse' not in stream_group or not isinstance(stream_group['latents_sparse'], h5py.Group):
                if self.verbose_load: print(f"Warning: Sparse latents not found in stream '{stream_name}'.")
                return None
                
            latents_sparse_group = stream_group['latents_sparse']
            batch_keys = sorted([key for key in latents_sparse_group.keys() if key.startswith('batch_')], 
                               key=lambda k: int(k.split('_')[1]))
            
            if not batch_keys:
                if self.verbose_load: print(f"Warning: No batch data found for sparse latents in stream '{stream_name}'.")
                return None
                
            first_batch_group = latents_sparse_group[batch_keys[0]]
            if 'original_shape' not in first_batch_group.attrs:
                if self.verbose_load: print(f"Warning: Missing shape information for sparse latents in stream '{stream_name}'.")
                return None
                
            original_shape = first_batch_group.attrs['original_shape']
            total_samples = sum(self.batch_lengths)
            
            if return_sparse:
                all_values = []
                all_row_indices = []
                all_col_indices = []
                current_offset = 0
                for batch_idx, batch_key in enumerate(batch_keys):
                    batch_group = latents_sparse_group[batch_key]
                    if not all(ds in batch_group for ds in ['values', 'batch_indices', 'latent_indices']):
                        if self.verbose_load: print(f"Warning: Missing datasets in batch {batch_key} for sparse latents.")
                        current_offset += self.batch_lengths[batch_idx]
                        continue
                        
                    values = batch_group['values'][:]
                    batch_indices = batch_group['batch_indices'][:]
                    latent_indices = batch_group['latent_indices'][:]
                    
                    # Convert local batch_indices to global row_indices
                    global_batch_indices = batch_indices + current_offset
                    
                    all_values.extend(values)
                    all_row_indices.extend(global_batch_indices)
                    all_col_indices.extend(latent_indices)
                    
                    current_offset += self.batch_lengths[batch_idx]
                
                if not all_values:
                     return csr_matrix((total_samples, original_shape[1]), dtype=np.float32)

                return csr_matrix((all_values, (all_row_indices, all_col_indices)),
                                  shape=(total_samples, original_shape[1]), dtype=np.float32)
            else:
                all_latents = np.zeros((total_samples, original_shape[1]), dtype=np.float32)
                
                current_offset = 0
                for batch_idx, batch_key in enumerate(batch_keys):
                    batch_group = latents_sparse_group[batch_key]
                    if not all(ds in batch_group for ds in ['values', 'batch_indices', 'latent_indices']):
                        if self.verbose_load: print(f"Warning: Missing datasets in batch {batch_key} for sparse latents.")
                        current_offset += self.batch_lengths[batch_idx]
                        continue
                        
                    values = batch_group['values'][:]
                    batch_indices = batch_group['batch_indices'][:]
                    latent_indices = batch_group['latent_indices'][:]
                    
                    for i, (local_b_idx, l_idx, val) in enumerate(zip(batch_indices, latent_indices, values)):
                        global_idx = current_offset + local_b_idx
                        if 0 <= global_idx < total_samples:
                            all_latents[global_idx, l_idx] = val
                    
                    current_offset += self.batch_lengths[batch_idx]
                    
                return all_latents
        
        if feature_type not in stream_group or not isinstance(stream_group[feature_type], h5py.Group):
            if self.verbose_load: print(f"Warning: Feature type '{feature_type}' not found in stream '{stream_name}'.")
            return None
        
        feature_group = stream_group[feature_type]
        all_feature_batches = _load_h5_group_as_list_of_arrays(feature_group)
        
        if not all_feature_batches:
            if self.verbose_load: print(f"Warning: No data batches found for '{feature_type}' in stream '{stream_name}'.")
            return None
        
        try:
            concatenated_features = np.concatenate(all_feature_batches, axis=0)
            return concatenated_features
        except ValueError as e:
            if self.verbose_load: print(f"Error concatenating features for '{feature_type}' in stream '{stream_name}': {e}")
            return None 

    def get_all_cross_reconstruction_features(self, cross_recon_key: str) -> Optional[np.ndarray]:
        """
        Retrieves all features for a given cross-reconstruction key, concatenated into a single NumPy array.
        Example key: 'clip_img_from_dino'.
        Returns None if the key is not found or has no data.
        """
        if 'cross_reconstructions' not in self.hf or \
           not isinstance(self.hf['cross_reconstructions'], h5py.Group):
            if self.verbose_load: print("Warning: 'cross_reconstructions' group not found in HDF5 file.")
            return None
        
        cr_base_group = self.hf['cross_reconstructions']
        if cross_recon_key not in cr_base_group or \
           not isinstance(cr_base_group[cross_recon_key], h5py.Group):
            if self.verbose_load: print(f"Warning: Cross-reconstruction key '{cross_recon_key}' not found.")
            return None
            
        feature_group = cr_base_group[cross_recon_key]
        all_feature_batches = _load_h5_group_as_list_of_arrays(feature_group)

        if not all_feature_batches:
            if self.verbose_load: print(f"Warning: No data batches found for cross-reconstruction key '{cross_recon_key}'.")
            return None
        
        try:
            concatenated_features = np.concatenate(all_feature_batches, axis=0)
            return concatenated_features
        except ValueError as e:
            if self.verbose_load: print(f"Error concatenating features for cross-reconstruction key '{cross_recon_key}': {e}")
            return None

    def get_cross_reconstruction_keys(self) -> List[str]:
        """Returns a list of available cross-reconstruction keys."""
        if 'cross_reconstructions' in self.hf and isinstance(self.hf['cross_reconstructions'], h5py.Group):
            return list(self.hf['cross_reconstructions'].keys())
        return []

    def has_feature_type(self, stream_name: str, feature_type: str) -> bool:
        """Checks if a specific feature type exists for a given stream."""
        if stream_name in self.hf and isinstance(self.hf[stream_name], h5py.Group):
            stream_group = self.hf[stream_name]
            return feature_type in stream_group and isinstance(stream_group[feature_type], h5py.Group)
        return False

    @property
    def n_latents(self) -> int:
        return self._n_latents

    def close(self):
        """Closes the HDF5 file."""
        if self.hf:
            try:
                self.hf.close()
                # Conditional print based on an instance attribute, if desired for quiet mode.
                if getattr(self, 'verbose_load', False):
                    print(f"Closed HDF5 analysis results file: {self.filepath}")
            except Exception as e:
                print(f"Error closing HDF5 file {self.filepath}: {e}")
            finally:
                self.hf = None

    def __del__(self):
        # Ensure file is closed when object is garbage collected, if not already closed.
        if self.hf is not None:
            self.close()

def get_analysis_results_h5(
    checkpoint_dir: str,
    config_dict: Dict, 
    model: torch.nn.Module,
    batch_size: int,
    seed: int,
    use_cache: bool = True,
    verbose: bool = False,
    cache_filename = "analysis_cache_val.h5"
) -> HDF5AnalysisResultsDataset:
    """
    Ensures analysis results HDF5 file is available and returns an HDF5AnalysisResultsDataset for lazy loading.
    If use_cache is True and the HDF5 file exists, it uses it.
    Otherwise, it computes the results, saves them incrementally to HDF5, and then provides the dataset.

    This function will raise appropriate exceptions if any step fails, rather than catching and returning None.
    """
    if 'test_stream_feature_files' in config_dict:
        val_feature_files = config_dict.get('test_stream_feature_files')
    else:
        val_feature_files = config_dict.get('val_stream_feature_files')
    if not val_feature_files:
        raise ValueError("'val_stream_feature_files' not found in config_dict.")

    cache_path = os.path.join(checkpoint_dir, cache_filename)

    # If cache is not used, or if it is used but file doesn't exist, we need to compute
    if not use_cache or (use_cache and not os.path.exists(cache_path)):
        if verbose:
            if not use_cache and os.path.exists(cache_path):
                print(f"'use_cache' is False. Recomputing and overwriting existing HDF5 cache: {cache_path}...")
            else:
                print(f"HDF5 cache not found or 'use_cache' is False. Running analysis and saving incrementally to: {cache_path}...")
        
        # Compute and save analysis results to HDF5 file
        # This will raise exceptions if it fails, which we want
        _compute_and_save_analysis_incrementally_to_h5(
            stream_feature_files=val_feature_files,
            model=model,
            output_h5_filepath=cache_path,
            batch_size=batch_size,
            seed=seed,
            verbose=verbose
        )
        if verbose: print(f"Analysis computation and HDF5 saving complete to {cache_path}.")

    # At this point, the HDF5 file should exist
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Expected HDF5 cache file {cache_path} to exist after processing, but it was not found.")
    
    if verbose: print(f"Providing HDF5AnalysisResultsDataset for: {cache_path}")
    
    # Create and return the dataset
    # This will raise exceptions if the file is invalid or corrupted
    return HDF5AnalysisResultsDataset(
        analysis_h5_filepath=cache_path,
        batch_size_used_for_generation=batch_size
    )
