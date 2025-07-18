import os, gc, h5py, torch, torchvision.transforms as T
import torchvision.datasets as dset
from pathlib import Path
from tqdm.auto import tqdm
import open_clip
import argparse                                     # ← NEW

# ─── helper to turn model names into safe file‑name tags ────────────────────
def tag(txt: str) -> str:
    return txt.replace('/', '_').replace(':', '-').replace('.', '-')

# ─── Data transformation helpers ────────────────────────────────────────────
class MaybeToTensor(T.ToTensor):
    def __call__(self, pic):
        return pic if isinstance(pic, torch.Tensor) else super().__call__(pic)

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
dino_transform = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    MaybeToTensor(),
    normalize,
])

class CocoImagesDataset(torch.utils.data.Dataset):
    def __init__(self, coco_ds, transform=None):
        self.coco_ds  = coco_ds
        self.transform = transform
    def __len__(self): return len(self.coco_ds)
    def __getitem__(self, idx):
        image, caps = self.coco_ds[idx]
        if self.transform: image = self.transform(image)
        # Ensure image is PIL for OpenCLIP if no transform applied
        if self.transform is None and not image.mode == 'RGB':
             image = image.convert('RGB')
        # Combine captions into a single string if needed
        captions_str = ' '.join(caps) if isinstance(caps, list) else caps
        return {'image': image, 'captions': captions_str, 'idx': idx}

def clip_collate(batch):
    return {
        'image'   : [b['image']    for b in batch],     # list[PIL.Image]
        'captions': [b['captions'] for b in batch],     # list[str]
        'idx'     : torch.tensor([b['idx'] for b in batch], dtype=torch.long),
    }

# ─── Feature Extraction Functions ───────────────────────────────────────────
def extract_dino_features_coco(coco_ds, args, device):
    print(f"Extracting DINO features for {args.split} split...")
    loader = torch.utils.data.DataLoader(
        CocoImagesDataset(coco_ds, dino_transform),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    model = torch.hub.load('facebookresearch/dinov2', args.dino_model).to(device).eval()
    dino_tag = tag(args.dino_model)
    out_path = args.out_dir / f'dino_features_{dino_tag}.h5'
    total_imgs = len(coco_ds)

    with h5py.File(out_path, 'w', libver='latest') as h5f:
        feat_ds = idx_ds = None
        for batch in tqdm(loader, desc=f'DINO ({dino_tag})'):
            imgs, idxs = batch['image'].to(device), batch['idx'].numpy()
            with torch.no_grad():
                feats = model(imgs).cpu().float()

            if feat_ds is None:
                fdim = feats.shape[1]
                feat_ds = h5f.create_dataset(
                    'features', (total_imgs, fdim), 'float32',
                    chunks=(min(args.batch_size, total_imgs), fdim), compression='gzip'
                )
                idx_ds = h5f.create_dataset('indices', (total_imgs,), 'int32')

            feat_ds[idxs, :] = feats.numpy()
            idx_ds[idxs] = idxs
            h5f.flush()

    del model; torch.cuda.empty_cache(); gc.collect()
    print(f"DINO features saved to {out_path}")

def extract_clip_features_coco(coco_ds, args, device):
    print(f"Extracting OpenCLIP features for {args.split} split...")
    clip_tag = tag(f'{args.clip_arch}-{args.clip_checkpt}')
    model, _, preprocess_clip = open_clip.create_model_and_transforms(
        args.clip_arch, pretrained=args.clip_checkpt)
    tokenizer = open_clip.get_tokenizer(args.clip_arch)
    model = model.to(device).eval()

    loader = torch.utils.data.DataLoader(
        CocoImagesDataset(coco_ds, transform=None), # Keep as PIL images
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=clip_collate,
    )

    img_path = args.out_dir / f'clip_image_features_{clip_tag}.h5'
    txt_path = args.out_dir / f'clip_text_features_{clip_tag}.h5'
    total_imgs = len(coco_ds)

    with h5py.File(img_path, 'w', libver='latest') as h5_img, \
         h5py.File(txt_path, 'w', libver='latest') as h5_txt:

        feat_img = feat_txt = idx_img = idx_txt = None
        for batch in tqdm(loader, desc=f'OpenCLIP ({clip_tag})'):
            pil_imgs, caps, idxs = batch['image'], batch['captions'], batch['idx'].numpy()

            img_tensor = torch.stack([preprocess_clip(im) for im in pil_imgs]).to(device)
            text_tokens = tokenizer(caps).to(device)

            with torch.no_grad(), torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                f_img = model.encode_image(img_tensor).cpu().float()
                f_txt = model.encode_text(text_tokens).cpu().float()

            if feat_img is None:
                dim_i, dim_t = f_img.shape[1], f_txt.shape[1]
                feat_img = h5_img.create_dataset('features', (total_imgs, dim_i), 'float32', chunks=(min(args.batch_size, total_imgs), dim_i), compression='gzip')
                idx_img = h5_img.create_dataset('indices', (total_imgs,), 'int32')
                feat_txt = h5_txt.create_dataset('features', (total_imgs, dim_t), 'float32', chunks=(min(args.batch_size, total_imgs), dim_t), compression='gzip')
                idx_txt = h5_txt.create_dataset('indices', (total_imgs,), 'int32')

            feat_img[idxs, :] = f_img.numpy(); idx_img[idxs] = idxs
            feat_txt[idxs, :] = f_txt.numpy(); idx_txt[idxs] = idxs
            h5_img.flush(); h5_txt.flush()

    del model; torch.cuda.empty_cache(); gc.collect()
    print(f"OpenCLIP image features saved to {img_path}")
    print(f"OpenCLIP text features saved to {txt_path}")

# ─── Main Execution ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Extract DINOv2 and OpenCLIP features for COCO.')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val'],
                        help='Dataset split to process (train or val)')
    parser.add_argument('--data_root', type=str, default='./dataset/COCO',
                        help='Root directory of the COCO dataset')
    parser.add_argument('--output_dir_base', type=str, default='./features/coco',
                        help='Base directory to save extracted features')
    parser.add_argument('--dino_model', type=str, default='dinov2_vitb14_reg',
                        help='DINOv2 model name')
    parser.add_argument('--clip_arch', type=str, default='ViT-B-16',
                        help='OpenCLIP model architecture')
    parser.add_argument('--clip_checkpt', type=str, default='datacomp_xl_s13b_b90k',
                        help='OpenCLIP pre-trained checkpoint name')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--extract_dino', action='store_true', help='Extract DINO features')
    parser.add_argument('--extract_clip', action='store_true', help='Extract OpenCLIP features')

    args = parser.parse_args()

    if not args.extract_dino and not args.extract_clip:
        print("Please specify at least one feature type to extract (--extract_dino or --extract_clip)")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Configure paths based on split
    data_root = Path(args.data_root)
    img_dir = data_root / f'{args.split}2017'
    ann_file = data_root / f'annotations/captions_{args.split}2017.json'
    args.out_dir = Path(args.output_dir_base) / args.split
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading COCO {args.split} captions from {ann_file}")
    coco_caps = dset.CocoCaptions(root=str(img_dir), annFile=str(ann_file))
    print(f"Found {len(coco_caps)} images in {args.split} split.")

    if args.extract_dino:
        extract_dino_features_coco(coco_caps, args, device)

    if args.extract_clip:
        extract_clip_features_coco(coco_caps, args, device)

    print("Feature extraction complete.")

if __name__ == "__main__":
    main()


