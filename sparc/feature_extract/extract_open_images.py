import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from glob import glob
import argparse
import os, gc, h5py, torch, torchvision.transforms as T
from tqdm.auto import tqdm
import open_clip                                            
from pathlib import Path


import os
import json
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset


class OpenImagesDataset(Dataset):
    def __init__(self, dataset_dir, split='train', transform=None, check_images=False):
        """
        Args:
            dataset_dir (str): Root directory for the dataset
            split (str): Dataset split ('train', 'test', 'validation')
            transform (callable, optional): Optional transform to be applied on images
            check_images (bool): Whether to check if images exist during initialization
        """
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform
        
        # Define paths
        self.img_dir = os.path.join(dataset_dir, 'images', split)
        self.cap_dir = os.path.join(dataset_dir, 'captions', split)
        self.labels_dir = os.path.join(dataset_dir, 'labels')
        
        # Find the simplified JSON file
        json_path = os.path.join(self.cap_dir, f'simplified_open_images_{split}_localized_narratives.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Simplified JSON file not found: {json_path}")
        
        # Load the caption data
        print(f"Loading caption data from {json_path}...")
        with open(json_path, 'r') as f:
            self.image_id_to_captions = json.load(f)
        
        # Load label data
        self._load_label_data()
        
        # Create a flat list of image_ids and caption indices for __getitem__
        self.samples = []
        for image_id, captions in self.image_id_to_captions.items():
            for caption_idx, _ in enumerate(captions):
                self.samples.append((image_id, caption_idx))
        
        if check_images:
            self._check_images()
    
    def _load_label_data(self):
        """Load class descriptions and annotations for the dataset."""
        print("Loading label data...")
        
        # Load class descriptions
        class_file = os.path.join(self.labels_dir, 'oidv7-class-descriptions-boxable.csv')
        if not os.path.exists(class_file):
            raise FileNotFoundError(f"Class descriptions file not found: {class_file}")
        
        class_df = pd.read_csv(class_file)
        self.label_to_class = {row['LabelName']: row['DisplayName'] 
                              for _, row in class_df.iterrows()}
        self.class_to_label = {v: k for k, v in self.label_to_class.items()}
        
        # Create ordered list of all classes for tensor conversion
        self.all_classes = sorted(list(self.label_to_class.values()))
        self.num_classes = len(self.all_classes)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.all_classes)}
        
        print(f"Total number of classes: {self.num_classes}")
        
        # Load annotations for the split
        annotations_file = os.path.join(self.labels_dir, 
                                       f'{self.split}-annotations-human-imagelabels-boxable.csv')
        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
        
        print(f"Loading annotations from {annotations_file}...")
        annotations_df = pd.read_csv(annotations_file)
        
        # Filter for positive labels (Confidence=1) if that column exists
        if 'Confidence' in annotations_df.columns:
            annotations_df = annotations_df[annotations_df['Confidence'] == 1]
        
        # Group labels by image ID
        self.image_to_labels = {}
        self.image_to_label_tensor = {}
        
        for image_id, group in annotations_df.groupby('ImageID'):
            label_names = group['LabelName'].tolist()
            # Convert label IDs to human-readable class names
            class_names = [self.label_to_class.get(label, label) for label in label_names]
            self.image_to_labels[image_id] = class_names
            
            # Create multi-hot encoding tensor for this image
            label_tensor = torch.zeros(self.num_classes, dtype=torch.float32)
            for class_name in class_names:
                if class_name in self.class_to_idx:
                    label_tensor[self.class_to_idx[class_name]] = 1.0
            
            self.image_to_label_tensor[image_id] = label_tensor
            
        print(f"Loaded labels for {len(self.image_to_labels)} images")

    def _check_images(self):
        """Check if all images in the dataset exist."""
        print("Checking image files...")
        unique_image_ids = set(image_id for image_id, _ in self.samples)
        
        valid_image_ids = set()
        for image_id in tqdm(unique_image_ids):
            image_path = os.path.join(self.img_dir, f"{image_id}.jpg")
            if os.path.isfile(image_path):
                valid_image_ids.add(image_id)
        
        # Filter samples to only include valid images
        self.samples = [(image_id, caption_idx) for image_id, caption_idx in self.samples 
                        if image_id in valid_image_ids]
        
        print(f"Found {len(valid_image_ids)} valid images out of {len(unique_image_ids)}")
        print(f"Dataset now contains {len(self.samples)} valid image-caption pairs")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict: Dictionary containing image tensor, caption string, image ID, and labels
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image_id and caption_idx
        image_id, caption_idx = self.samples[idx]
        
        # Get caption data
        caption_data = self.image_id_to_captions[image_id][caption_idx]
        caption = caption_data['caption']
        
        # Get labels for this image (both text and tensor versions)
        labels_text = self.image_to_labels.get(image_id, [])
        
        # Get or create the multi-hot encoded label tensor
        if image_id in self.image_to_label_tensor:
            labels_tensor = self.image_to_label_tensor[image_id]
        else:
            # If image has no labels in our dataset, create empty tensor
            labels_tensor = torch.zeros(self.num_classes, dtype=torch.float32)
        
        # Load image
        image_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'captions': caption,
            'image_id': image_id,
            'labels_text': labels_text,
            'labels': labels_tensor,  # This is the multi-hot encoded tensor
            'idx': idx
        }


def extract_dino_features_open_images(dataset_dir, split, OUT_DIR, DINO_MODEL, DINO_TAG, BATCH_SIZE, DEVICE):
    loader = torch.utils.data.DataLoader(
        OpenImagesDataset(dataset_dir, split,  dino_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    TOTAL_IMGS = len(loader.dataset)
    model = torch.hub.load('facebookresearch/dinov2', DINO_MODEL).to(DEVICE).eval()

    out_path = OUT_DIR / f'dino_features_{DINO_TAG}.h5'
    with h5py.File(out_path, 'w', libver='latest') as h5f:
        feat_ds = idx_ds = None
        for batch in tqdm(loader, desc=f'DINO ({DINO_TAG})'):
            imgs, idxs = batch['image'].to(DEVICE), batch['idx'].numpy()
            with torch.no_grad():
                feats = model(imgs).cpu().float()

            if feat_ds is None:                      # first batch → allocate
                fdim = feats.shape[1]
                feat_ds = h5f.create_dataset(
                    'features', (TOTAL_IMGS, fdim), 'float32',
                    chunks=(min(BATCH_SIZE, TOTAL_IMGS), fdim), compression='gzip'
                )
                idx_ds  = h5f.create_dataset('indices', (TOTAL_IMGS,), 'int32')

            feat_ds[idxs, :] = feats.numpy()
            idx_ds[idxs]     = idxs
            h5f.flush()

    del model; torch.cuda.empty_cache(); gc.collect()

def clip_collate(batch):
    return {
        'image'   : [b['image']    for b in batch],     # list[PIL.Image]
        'captions': [b['captions'] for b in batch],     # list[str]
        'idx'     : torch.tensor([b['idx'] for b in batch], dtype=torch.long),
    }

def extract_clip_features_open_images(dataset_dir, split, OUT_DIR, CLIP_MODEL, CLIP_TAG, BATCH_SIZE, DEVICE):
    # ── load model + transforms once ─────────────────────────────────────────    

    model, _, preprocess_clip = open_clip.create_model_and_transforms(CLIP_MODEL)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)

    model = model.to(DEVICE).eval()

    # ── DataLoader: keep PIL images & raw strings (collate already defined) ──
    loader = torch.utils.data.DataLoader(
        OpenImagesDataset(dataset_dir, split),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
        collate_fn=clip_collate,
    )
    TOTAL_IMGS = len(loader.dataset)
    # ── output paths ────────────────────────────────────────────────────────
    img_path = OUT_DIR / f'clip_image_features_{CLIP_TAG}.h5'
    txt_path = OUT_DIR / f'clip_text_features_{CLIP_TAG}.h5'

    with h5py.File(img_path, 'w', libver='latest') as h5_img, \
         h5py.File(txt_path, 'w', libver='latest') as h5_txt:

        feat_img = feat_txt = idx_img = idx_txt = None  # allocate on first batch

        for batch in tqdm(loader, desc=f'OpenCLIP ({CLIP_TAG})'):
            pil_imgs, caps, idxs = batch['image'], batch['captions'], batch['idx'].numpy()

            # preprocess ⇢ tensor ↦ device
            img_tensor  = torch.stack([preprocess_clip(im) for im in pil_imgs]).to(DEVICE)
            text_tokens = tokenizer(caps).to(DEVICE)

            with torch.no_grad(), torch.autocast(device_type='cuda',
                                                 enabled=(DEVICE.type == 'cuda')):
                f_img = model.encode_image(img_tensor).cpu().float()
                f_txt = model.encode_text(text_tokens).cpu().float()

            if feat_img is None:                         # first pass → create datasets
                dim_i, dim_t = f_img.shape[1], f_txt.shape[1]

                feat_img = h5_img.create_dataset(
                    'features', (TOTAL_IMGS, dim_i), 'float32',
                    chunks=(min(BATCH_SIZE, TOTAL_IMGS), dim_i), compression='gzip')
                idx_img  = h5_img.create_dataset('indices', (TOTAL_IMGS,), 'int32')

                feat_txt = h5_txt.create_dataset(
                    'features', (TOTAL_IMGS, dim_t), 'float32',
                    chunks=(min(BATCH_SIZE, TOTAL_IMGS), dim_t), compression='gzip')
                idx_txt  = h5_txt.create_dataset('indices', (TOTAL_IMGS,), 'int32')

            feat_img[idxs, :] = f_img.numpy();  idx_img[idxs] = idxs
            feat_txt[idxs, :] = f_txt.numpy();  idx_txt[idxs] = idxs
            h5_img.flush();  h5_txt.flush()

    del model; torch.cuda.empty_cache(); gc.collect()
    
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

class MaybeToTensor(T.ToTensor):
    def __call__(self, pic):
        return pic if isinstance(pic, torch.Tensor) else super().__call__(pic)

normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
dino_transform = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    MaybeToTensor(),
    normalize,
])

if __name__ == "__main__":
    DINO_MODEL     = 'dinov2_vitl14_reg'          
    CLIP_MODEL     = 'hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K'#'hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K'
    DINO_TAG     = 'dinov2_vitl14_reg'          
    CLIP_TAG     = 'CLIP-ViT-L-14-DataComp'
    BATCH_SIZE     = 32
    DATASET_DIR    = '/home/atlas-gp/OpenImages/'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = argparse.ArgumentParser()
    args.add_argument('--model', choices=['dinov2', 'clip'], required=True)
    args.add_argument('--split', choices=['train', 'test', 'val'], required=True)
    args = args.parse_args()

    OUT_DIR = Path(f'./features/open_images/{args.split}')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.model == 'dinov2':
        extract_dino_features_open_images(DATASET_DIR, args.split, OUT_DIR, DINO_MODEL, DINO_TAG, BATCH_SIZE, DEVICE)
    elif args.model == 'clip':
        extract_clip_features_open_images(DATASET_DIR, args.split, OUT_DIR, CLIP_MODEL, CLIP_TAG, BATCH_SIZE, DEVICE)
