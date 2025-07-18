import torch
import torchvision.datasets as dset
from pycocotools.coco import COCO
import os
import json
from tqdm import tqdm
from ..utils import seed_worker, set_seed
import pandas as pd
from pathlib import Path
import abc # Added for Abstract Base Class
from typing import Optional
import torch.nn as nn
import ast
from sklearn.metrics import average_precision_score, accuracy_score, recall_score, precision_score, f1_score
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import numpy as np

from ..post_analysis import HDF5AnalysisResultsDataset
from ..feature_extract.extract_open_images import OpenImagesDataset as OpenImagesFeatureDataset


# --- Abstract Base Class for Label and Metadata Provision ---
class BaseLabelProvider(abc.ABC):
    @abc.abstractmethod
    def get_num_classes(self) -> int:
        pass

    @abc.abstractmethod
    def get_cat_names(self) -> dict: # Maps internal index 0..N-1 to label string
        pass

    @abc.abstractmethod
    def get_label_tensor(self, item_original_idx: int) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_label_to_idx_map(self) -> dict: # Maps original label (e.g. COCO cat_id) to internal index
        pass

    @abc.abstractmethod
    def get_idx_to_label_map(self) -> dict: # Maps internal index to original label
        pass

# --- Concrete Implementations for Each Dataset Type ---

class CocoLabelProvider(BaseLabelProvider):
    def __init__(self, coco_root_folder: str, verbose: bool = False):
        if verbose: print("Initializing CocoLabelProvider...")
        val_img_dir = os.path.join(coco_root_folder, 'val2017')
        val_ann_file_caps = os.path.join(coco_root_folder, 'annotations/captions_val2017.json')
        val_ann_file_instances = os.path.join(coco_root_folder, 'annotations/instances_val2017.json')
        if not all(os.path.exists(p) for p in [val_img_dir, val_ann_file_caps, val_ann_file_instances]):
            raise FileNotFoundError(f"COCO validation data/annotations not found in {coco_root_folder}.")
        
        self.coco_caps = dset.CocoCaptions(root=val_img_dir, annFile=val_ann_file_caps)
        self.coco = COCO(val_ann_file_instances)
        if verbose: print(f"  Loaded COCO annotations. Found {len(self.coco.getImgIds())} images.")

        categories = self.coco.loadCats(self.coco.getCatIds())
        self._cat_ids = self.coco.getCatIds()
        self._num_classes = len(self._cat_ids)
        
        # Mapping from COCO category ID to internal sequential index (0 to N-1)
        self._label_to_idx_map = {cat_id: i for i, cat_id in enumerate(self._cat_ids)}
        # Mapping from internal sequential index to COCO category ID
        self._idx_to_label_map = {i: cat_id for i, cat_id in enumerate(self._cat_ids)}
        
        cat_names_by_id = {cat['id']: cat['name'] for cat in categories}
        # self._cat_names maps internal index 0..N-1 to label string (category name)
        self._cat_names = {idx: cat_names_by_id[cat_id] for idx, cat_id in self._idx_to_label_map.items()}

    def get_num_classes(self) -> int:
        return self._num_classes

    def get_cat_names(self) -> dict:
        return self._cat_names

    def get_label_to_idx_map(self) -> dict:
        return self._label_to_idx_map

    def get_idx_to_label_map(self) -> dict:
        return self._idx_to_label_map

    def get_label_tensor(self, item_original_idx: int) -> torch.Tensor:
        label_tensor = torch.zeros(self._num_classes, dtype=torch.float32)
        img_id = self.coco_caps.ids[item_original_idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id in self._label_to_idx_map:
                label_tensor[self._label_to_idx_map[cat_id]] = 1.0
        return label_tensor

class OpenImagesLabelProvider(BaseLabelProvider):
    def __init__(self, open_images_root_folder: str, split: str, verbose: bool = False):
        if verbose: print(f"Initializing OpenImagesLabelProvider for split '{split}'...")
        # Direct import at top ensures OpenImagesFeatureDataset is available or import fails early.
        self.oi_dataset = OpenImagesFeatureDataset(dataset_dir=open_images_root_folder, split=split, transform=None, check_images=False)
        
        self._num_classes = self.oi_dataset.num_classes
        self._cat_names = {idx: name for idx, name in enumerate(self.oi_dataset.all_classes)}
        self._label_to_idx_map = self.oi_dataset.class_to_idx
        self._idx_to_label_map = {idx: name for name, idx in self._label_to_idx_map.items()}
        if verbose: print(f"  OpenImages: Found {self._num_classes} classes.")

    def get_num_classes(self) -> int:
        return self._num_classes

    def get_cat_names(self) -> dict:
        return self._cat_names

    def get_label_to_idx_map(self) -> dict:
        return self._label_to_idx_map
    
    def get_idx_to_label_map(self) -> dict:
        return self._idx_to_label_map

    def get_label_tensor(self, item_original_idx: int) -> torch.Tensor:
        # item_original_idx is the integer index from the original dataset enumeration (0 to N-1)
        # This will raise IndexError if item_original_idx is out of bounds for self.oi_dataset.samples
        image_id_str, _ = self.oi_dataset.samples[item_original_idx]
        if image_id_str in self.oi_dataset.image_to_label_tensor:
            return self.oi_dataset.image_to_label_tensor[image_id_str]
        else:
            return torch.zeros(self._num_classes, dtype=torch.float32)

# --- Helper Functions ---
def save_classification_results(evaluation_metrics, args, wandb_run=None):
    results_path = os.path.join(args.checkpoint_dir, "classification_results.json")
    if args.verbose: print(f"Saving evaluation results to: {results_path}")
    with open(results_path, 'w') as f:
        json.dump(evaluation_metrics, f, indent=4)

    summary_metrics = {}
    for target_stream, stream_metrics in evaluation_metrics.items():
        for metric_key, results in stream_metrics.items():
            if isinstance(results, dict) and 'test_mAP' in results:
                summary_metrics[metric_key] = results['test_mAP']
            # else: # Removed warning for missing test_mAP, assume it might be validly missing if a classifier failed
                  # or if a metric type doesn't produce mAP. The summary will just not include it.

    summary_path = os.path.join(args.checkpoint_dir, "classification_summary.json")
    if args.verbose: print(f"Saving evaluation summary to: {summary_path}")
    with open(summary_path, 'w') as f:
        json.dump(summary_metrics, f, indent=4)
        
    if wandb_run and summary_metrics:
        if args.verbose: print("Logging classification summary to Wandb (using log)...")
        # Let WandB logging errors propagate
        wandb_classification_log = {f"classification/{k}": v for k, v in summary_metrics.items()}
        wandb_run.log(wandb_classification_log)
    return summary_metrics


def concat_batches_to_tensor(batch_list):
    if not batch_list: return torch.empty(0) # This might be a valid case if a loader is empty.
                                           # If it's an error, subsequent code should fail.
    concatenated_np = np.concatenate(batch_list, axis=0)
    return torch.from_numpy(concatenated_np).float()

# Dataset of latent representations with labels
class LatentDataset(Dataset):
    def __init__(self, latents, indices, label_provider: BaseLabelProvider):
        self.latents = latents
        self.indices = indices # These are the original dataset indices/IDs
        self.label_provider = label_provider

        self.num_classes = self.label_provider.get_num_classes()
        self.cat_names = self.label_provider.get_cat_names() # For reporting

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        latent = self.latents[idx]
        item_original_idx = self.indices[idx].item() # Get the original ID for this sample
        
        label_tensor = self.label_provider.get_label_tensor(item_original_idx)
        
        return latent, label_tensor, item_original_idx

def mean_average_precision(y_true, y_score):
    mask = y_true.sum(axis=0) > 0          # keep only classes that appear
    mAP_macro_no_empty = average_precision_score(
        y_true[:, mask], y_score[:, mask], average="macro"
    )
    return mAP_macro_no_empty

class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=80):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # return torch.sigmoid(self.classifier(x))
        return self.classifier(x)
    

def eval_classification(analysis_results: HDF5AnalysisResultsDataset, args, config_dict, wandb_run=None, verbose=False):
    if args.verbose: print("--- Starting Classification Evaluation (HDF5 Mode) ---")
    set_seed(args.seed)
    split_generator = torch.Generator()
    split_generator.manual_seed(args.seed)

    label_provider: Optional[BaseLabelProvider] = None
    dataset_name_from_config = config_dict.get('dataset', '').lower()

    all_original_indices = analysis_results.get_all_original_dataset_indices()
    if all_original_indices is None or len(all_original_indices) == 0:
        analysis_results.close()
        raise ValueError("No dataset indices found in HDF5 analysis results. Cannot proceed with classification.")

    if 'coco_root_folder' in config_dict and (not dataset_name_from_config or dataset_name_from_config == 'coco'):
        if args.verbose: print("Detected COCO dataset configuration.")
        label_provider = CocoLabelProvider(
            coco_root_folder=config_dict['coco_root_folder'],
            verbose=args.verbose
        )
    elif 'open_images_root_folder' in config_dict and (not dataset_name_from_config or dataset_name_from_config == 'open_images'):
        if args.verbose: print("Detected OpenImages dataset configuration.")
        open_images_split = 'val' 
        if 'test_stream_feature_files' in config_dict and config_dict['test_stream_feature_files']:
            open_images_split = 'test'
            if args.verbose: print(f"  OpenImages using 'test' split for labels based on config.")
        elif 'val_stream_feature_files' in config_dict and config_dict['val_stream_feature_files']:
            open_images_split = 'val'
            if args.verbose: print(f"  OpenImages using 'val' split for labels based on config.")
        else:
            analysis_cache_name = os.path.basename(analysis_results.filepath) 
            if "test" in analysis_cache_name:
                 open_images_split = 'test'
                 if args.verbose: print(f"  OpenImages using 'test' split for labels based on HDF5 filename '{analysis_cache_name}'.")
            else: 
                 open_images_split = 'val'
                 if args.verbose: print(f"  OpenImages using 'val' split for labels (default or based on HDF5 filename '{analysis_cache_name}'.")
        label_provider = OpenImagesLabelProvider(
            open_images_root_folder=config_dict['open_images_root_folder'],
            split=open_images_split,
            verbose=args.verbose
        )
    else:
        analysis_results.close()
        raise ValueError(
            "Could not determine dataset type for classification labels. "
            "Config missing 'coco_root_folder', "
            "or 'open_images_root_folder', or 'dataset' field in config does not match available options."
        )

    if args.verbose: print(f"--- Training Downstream Classifiers (HDF5 Mode) ---")
    evaluation_metrics = {}
    
    streams = analysis_results.streams
    if not streams:
        analysis_results.close()
        raise ValueError("No streams found in HDF5 analysis results. Cannot train classifiers.")

    if args.train_ratio is None or args.hidden_dim is None: 
        analysis_results.close()
        raise ValueError("Missing --train_ratio or --hidden_dim argument, required for classification.")
         
    feature_types_to_eval = ['raw', 'latents', 'recon']
    cross_recon_keys = analysis_results.get_cross_reconstruction_keys()
    
    total_classifiers = 0
    for stream in streams:
        for f_type in feature_types_to_eval:
            if analysis_results.has_feature_type(stream, f_type):
                 total_classifiers += 1
    total_classifiers += len(cross_recon_keys)
    
    if total_classifiers == 0:
        analysis_results.close()
        raise ValueError("No features found to train classifiers on in the HDF5 analysis results.")

    if args.verbose: print(f"Total classifiers to train: {total_classifiers}")
    pbar = tqdm(total=total_classifiers, desc="Training Classifiers", disable=not args.verbose)

    for target_stream in streams:
        evaluation_metrics[target_stream] = {}
        
        for f_type in feature_types_to_eval:
            if not analysis_results.has_feature_type(target_stream, f_type):
                continue # This stream/f_type combination doesn't exist, skip it.
            
            metric_key = f"{f_type}_{target_stream}"
            if args.verbose: print(f"\nProcessing {metric_key}...")

            current_features = analysis_results.get_all_features_for_stream(target_stream, f_type)
            if current_features is None or len(current_features) == 0:
                pbar.update(1)
                analysis_results.close()
                raise ValueError(f"No features found for {metric_key}, but has_feature_type was true. Data inconsistency.")
            
            if len(current_features) != len(all_original_indices):
                pbar.update(1)
                analysis_results.close()
                raise ValueError(f"Mismatch in feature count ({len(current_features)}) and index count ({len(all_original_indices)}) for {metric_key}.")

            metric = run_downstream_classification_eval(
                inputs=current_features, 
                indices=all_original_indices, 
                label_provider=label_provider,
                split_generator=split_generator,
                seed_worker=seed_worker,
                train_ratio=args.train_ratio,
                hidden_dim=args.hidden_dim,
                verbose=verbose 
            )
            # run_downstream_classification_eval now raises exceptions on failure
            evaluation_metrics[target_stream][metric_key] = metric
                 
            if args.verbose: pbar.set_description(f"Trained {metric_key}")
            pbar.update(1)

        for cross_recon_key in cross_recon_keys:
            if cross_recon_key.startswith(target_stream + "_from_"):
                metric_key = f"cross_recon_{cross_recon_key}"
                if args.verbose: print(f"\nProcessing {metric_key}...")

                current_cross_features = analysis_results.get_all_cross_reconstruction_features(cross_recon_key)
                if current_cross_features is None or len(current_cross_features) == 0:
                    pbar.update(1)
                    analysis_results.close()
                    raise ValueError(f"No features found for cross-reconstruction {metric_key}.")
                
                if len(current_cross_features) != len(all_original_indices):
                    pbar.update(1)
                    analysis_results.close()
                    raise ValueError(f"Mismatch in feature count ({len(current_cross_features)}) and index count ({len(all_original_indices)}) for {metric_key}.")

                metric = run_downstream_classification_eval(
                    inputs=current_cross_features, 
                    indices=all_original_indices,  
                    label_provider=label_provider,
                    split_generator=split_generator,
                    seed_worker=seed_worker,
                    train_ratio=args.train_ratio,
                    hidden_dim=args.hidden_dim,
                    verbose=False
                )
                evaluation_metrics[target_stream][metric_key] = metric
                
                if args.verbose: pbar.set_description(f"Trained {metric_key}")
                pbar.update(1)

    pbar.close()
    analysis_results.close() 
    summary_metrics = save_classification_results(evaluation_metrics, args, wandb_run)
    return summary_metrics

 
def run_downstream_classification_eval(
    inputs: np.ndarray, 
    indices: np.ndarray, 
    label_provider: BaseLabelProvider,
    split_generator=None,
    seed_worker=None,
    train_ratio=0.8, 
    hidden_dim=256, 
    verbose=False
):
    assert 0 < train_ratio < 1, "train_ratio must be between 0 and 1"
    if verbose:
        print(f"Running downstream classification: Input features shape {inputs.shape}, Indices shape {indices.shape}")
    
    latents = np.concatenate(inputs, axis=0) if isinstance(inputs, list) and inputs and all(isinstance(i, np.ndarray) for i in inputs) else inputs
    indices_arr = np.concatenate(indices, axis=0) if isinstance(indices, list) and indices and all(isinstance(i, np.ndarray) for i in indices) else indices

    if not isinstance(latents, np.ndarray) or not isinstance(indices_arr, np.ndarray):
        raise TypeError(f"Inputs (type: {type(latents)}) and indices (type: {type(indices_arr)}) must be NumPy arrays.")

    if latents.ndim == 1: 
        if latents.shape[0] > 0:
            # This case is highly unlikely to be correct for classification features.
            raise ValueError(f"Input latents are 1D ({latents.shape}), which is generally unsuitable for classification. Reshaping not performed to ensure intent.")
        else: 
            raise ValueError("Input latents are 1D and empty. Cannot train classifier.")
    elif latents.ndim != 2:
        raise ValueError(f"Input latents have unexpected dimension {latents.ndim} (shape {latents.shape}). Classifier expects 2D.")
    
    if latents.shape[0] == 0:
        raise ValueError("Input latents are empty (0 samples). Cannot train classifier.")
    if latents.shape[0] != indices_arr.shape[0]:
        raise ValueError(f"Mismatch between number of latent samples ({latents.shape[0]}) and indices ({indices_arr.shape[0]}).")

    # LatentDataset instantiation can raise ValueError if label_provider has issues (e.g. 0 classes after filtering)
    # Let such errors propagate.
    latent_dataset = LatentDataset(latents, indices_arr, label_provider)
    
    if len(latent_dataset) == 0:
        raise ValueError("LatentDataset is empty after initialization. Cannot train classifier.")
    if latent_dataset.num_classes == 0:
        raise ValueError("LatentDataset has 0 classes after initialization (e.g. no valid labels found by provider). Cannot train classifier.")

    if split_generator is None:
        split_generator = torch.Generator() # Use a default if not provided.
    if seed_worker is None:
        pass # Reproducibility might be affected but not a fatal error.

    train_size = int(train_ratio * len(latent_dataset))
    test_size = len(latent_dataset) - train_size
    if train_size == 0 or test_size == 0:
        raise ValueError(f"Not enough data for train/test split (train: {train_size}, test: {test_size}). Min 1 sample needed for each.")
    train_dataset, test_dataset = random_split(latent_dataset, [train_size, test_size], generator=split_generator)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, 
                              worker_init_fn=seed_worker, generator=split_generator)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, worker_init_fn=seed_worker, 
                             generator=split_generator)

    input_dim = latents.shape[1]
    num_classes = latent_dataset.num_classes
    
    def init_weights(m, generator):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1.0, generator=generator)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    classifier_model = MultiLabelClassifier(input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    weight_init_generator = torch.Generator().manual_seed(split_generator.initial_seed()) 
    classifier_model.apply(lambda m: init_weights(m, weight_init_generator))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier_model.to(device)

    temp_train_labels = []
    for i in range(len(train_dataset)):
        _, label_tensor, _ = train_dataset[i] 
        temp_train_labels.append(label_tensor)

    if not temp_train_labels:
         raise ValueError("No labels found in the training set for pos_weight calculation. Training cannot proceed.")
    
    label_counts = torch.stack(temp_train_labels).sum(dim=0)
    pos_weight = (len(train_dataset) - label_counts) / (label_counts + 1e-6) 
    pos_weight = torch.clamp(pos_weight, min=0.1, max=10.0) 
        
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.Adam(classifier_model.parameters(), lr=0.001)
    num_epochs = 15 # Example, should be configurable
    if verbose: print(f"  Training classifier for {num_epochs} epochs...")

    best_val_map = -1.0
    epochs_no_improve = 0
    patience = 15 

    for epoch in range(num_epochs):
        classifier_model.train()
        train_loss_epoch = 0.0
        train_iterator = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{num_epochs} (Train)", disable=not verbose, leave=False)
        for batch_latents, batch_labels, _ in train_iterator:
            batch_latents, batch_labels = batch_latents.to(device), batch_labels.to(device)
            outputs = classifier_model(batch_latents)
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item() * batch_latents.size(0)
        train_loss_epoch /= len(train_loader.dataset)

        classifier_model.eval()
        val_loss_epoch = 0.0
        all_val_preds_epoch = []
        all_val_labels_epoch = []
        val_iterator = tqdm(test_loader, desc=f"  Epoch {epoch+1}/{num_epochs} (Val)", disable=not verbose, leave=False)
        with torch.no_grad():
            for batch_latents, batch_labels, _ in val_iterator:
                batch_latents, batch_labels = batch_latents.to(device), batch_labels.to(device)
                outputs = classifier_model(batch_latents)
                loss = criterion(outputs, batch_labels)
                val_loss_epoch += loss.item() * batch_latents.size(0)
                all_val_preds_epoch.append(torch.sigmoid(outputs).cpu().numpy())
                all_val_labels_epoch.append(batch_labels.cpu().numpy())
        val_loss_epoch /= len(test_loader.dataset)
        
        all_val_preds_np = np.concatenate(all_val_preds_epoch, axis=0)
        all_val_labels_np = np.concatenate(all_val_labels_epoch, axis=0)
        val_map_epoch = mean_average_precision(all_val_labels_np, all_val_preds_np)

        if verbose:
            print(f"    Epoch {epoch+1}: Train Loss={train_loss_epoch:.4f}, Val Loss={val_loss_epoch:.4f}, Val mAP={val_map_epoch:.4f}")

        if val_map_epoch > best_val_map:
            best_val_map = val_map_epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            if verbose: print(f"  Early stopping triggered at epoch {epoch+1} with Val mAP: {best_val_map:.4f}")
            break

    if verbose: print("  Evaluating final model on test set...")
    classifier_model.eval()
    all_final_preds_sig = []
    all_final_labels = []
    all_final_scores_logits = []

    test_iterator_final = tqdm(test_loader, desc="  Final Test Eval", disable=not verbose, leave=False)
    with torch.no_grad():
        for batch_latents, batch_labels, _ in test_iterator_final:
            batch_latents, batch_labels = batch_latents.to(device), batch_labels.to(device)
            outputs_logits = classifier_model(batch_latents)
            outputs_sigmoids = torch.sigmoid(outputs_logits)
            all_final_preds_sig.append(outputs_sigmoids.cpu().numpy())
            all_final_labels.append(batch_labels.cpu().numpy())
            all_final_scores_logits.append(outputs_logits.cpu().numpy())

    all_final_preds_np = np.concatenate(all_final_preds_sig, axis=0)
    all_final_labels_np = np.concatenate(all_final_labels, axis=0)
    final_preds_binary = (all_final_preds_np > 0.5).astype(int)

    accuracy = accuracy_score(all_final_labels_np.flatten(), final_preds_binary.flatten())
    precision = precision_score(all_final_labels_np, final_preds_binary, average='samples', zero_division=0)
    recall = recall_score(all_final_labels_np, final_preds_binary, average='samples', zero_division=0)
    f1 = f1_score(all_final_labels_np, final_preds_binary, average='samples', zero_division=0)
    mAP = mean_average_precision(all_final_labels_np, all_final_preds_np)
    
    metrics = {
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'test_mAP': mAP,
        'best_val_mAP_early_stopped': best_val_map if epochs_no_improve >= patience else val_map_epoch
    }
    if verbose:
        print(f"  Final Evaluation Metrics:")
        for k, v in metrics.items(): print(f"    {k}: {v:.4f}")

    class_aps = []
    for i in range(all_final_labels_np.shape[1]): # Iterate through classes
        labels_i = all_final_labels_np[:, i]
        preds_i = all_final_preds_np[:, i]
        class_name = latent_dataset.cat_names.get(i, f"Unknown Class {i}")
        
        # Only calculate AP if there are positive samples for this class, otherwise sklearn warns/errors.
        ap = average_precision_score(labels_i, preds_i) if labels_i.sum() > 0 else np.nan
        class_aps.append((class_name, ap))
    
    # Filter out NaN APs before sorting
    sorted_class_aps = sorted([item for item in class_aps if not np.isnan(item[1])], key=lambda x: x[1], reverse=True)
    
    metrics['per_class_AP'] = {name: ap for name, ap in class_aps} # Store all, including NaNs for completeness
    metrics['per_class_AP_sorted'] = sorted_class_aps # Store sorted list of valid APs

    if verbose:
        print("  Per-class Average Precision (Top 5 valid):")
        for class_name, ap_val in sorted_class_aps[:5]: print(f"    {class_name}: {ap_val:.4f}")
        if len(sorted_class_aps) > 5:
            print("  Per-class Average Precision (Bottom 5 valid):")
            for class_name, ap_val in sorted_class_aps[-5:]: print(f"    {class_name}: {ap_val:.4f}")
    return metrics
