import h5py
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Tuple, Dict, Union


# --- Multi-Stream HDF5 Feature Dataset ---
class HDF5FeatureDataset(Dataset):
    """Dataset for loading features from multiple HDF5 files, one per stream."""

    def __init__(self, stream_files: Dict[str, str], return_index: bool = False):
        """
        Args:
            stream_files: Dictionary mapping stream names (e.g., 'dino', 'clip_img')
                            to paths of their corresponding HDF5 files.
            return_index: If True, __getitem__ returns (feature_dict, index).
                         Defaults to False.
        """
        self.stream_files = stream_files
        self.streams = list(stream_files.keys())
        self.return_index = return_index
        
        self.h5_handles: Dict[str, h5py.File] = {}
        self.features: Dict[str, h5py.Dataset] = {}
        self.n_samples: Dict[str, int] = {}
        self.d_streams: Dict[str, int] = {}

        try:
            for stream_name, file_path in self.stream_files.items():
                print(f"Loading {stream_name} features from: {file_path}")
                h5_file = h5py.File(Path(file_path), 'r')
                self.h5_handles[stream_name] = h5_file
                
                if 'features' not in h5_file:
                     raise ValueError(f"HDF5 file {file_path} for stream {stream_name} does not contain a 'features' dataset.")
                     
                self.features[stream_name] = h5_file['features']
                self.n_samples[stream_name] = self.features[stream_name].shape[0]
                self.d_streams[stream_name] = self.features[stream_name].shape[1]
                print(f"  {stream_name.upper()}: {self.n_samples[stream_name]} samples, dimension {self.d_streams[stream_name]}")

        except Exception as e:
            print(f"Error opening or reading HDF5 files: {e}")
            self.close() # Ensure any opened files are closed on error
            raise

        # --- Sanity Checks ---
        # Check if all streams have the same number of samples
        num_samples_set = set(self.n_samples.values())
        if len(num_samples_set) > 1:
            error_msg = f"Mismatch in number of samples across streams: {self.n_samples}"
            self.close()
            raise ValueError(error_msg)
        
        # Check if any files were actually loaded
        if not self.n_samples:
            raise ValueError("No valid stream files provided or loaded.")
            
        # Set the common dataset size
        self.N = list(num_samples_set)[0] 
        print(f"=> All ({len(self.streams)}) streams have {self.N} samples.")

    def __len__(self):
        return self.N

    def __getitem__(self, idx) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], int]]:
        # Read features for the index from all files
        feature_dict = {}
        for stream_name in self.streams:
            x = self.features[stream_name][idx].astype(np.float32)
            
            # Normalize each feature vector to unit norm individually
            norm = np.linalg.norm(x)
            if norm > 0: 
                x = x / (norm + 1e-8)
            feature_dict[stream_name] = x

        if self.return_index:
            return feature_dict, idx
        else:
            return feature_dict

    def get_feature_dims(self) -> Dict[str, int]:
        """Return the feature dimensions as a dictionary {stream_name: dim}."""
        return self.d_streams

    def close(self):
        """Close all opened HDF5 files."""
        files_closed_count = 0
        for stream_name, handle in self.h5_handles.items():
            if handle:
                try:
                    handle.close()
                    files_closed_count += 1
                except Exception as e:
                    print(f"Error closing HDF5 file handle for stream {stream_name}: {e}")
        if files_closed_count > 0:
             print(f"Closed {files_closed_count} HDF5 file(s).")

        # Clear handles after closing
        self.h5_handles.clear()

    def __del__(self):
        self.close()