#!/usr/bin/env python3
"""
Caption-based DINO Heatmap Visualization Script
Generates visualizations for image-caption pairs using DINO and CLIP models with MS-SAE.
Based on the t3_dino_heatmaps_from_caption_open_imgs.ipynb notebook.
"""

import os
import gc
import torch
import torchvision.transforms as T
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import open_clip
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

# Import custom modules
from sparc.heatmaps.attention_relevance import interpret_sparc, interpret_clip
from sparc.heatmaps.gradcam import compute_gradcam
from sparc.heatmaps.clip import create_wrapped_clip
from sparc.heatmaps.dino import create_wrapped_dinov2
from sparc.model.model_global import MultiStreamSparseAutoencoder as MSAE_Global
from sparc.feature_extract.extract_open_images import OpenImagesDataset

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DINO_MODEL = 'dinov2_vitl14_reg'          
CLIP_MODEL = 'hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K'
DINO_TAG = 'dinov2_vitl14_reg'          
CLIP_TAG = 'CLIP-ViT-L-14-DataComp'

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

class MaybeToTensor(T.ToTensor):
    def __call__(self, pic):
        return pic if isinstance(pic, torch.Tensor) else super().__call__(pic)

def _text_size(f, txt):
    if hasattr(f, "getbbox"):
        x0, y0, x1, y1 = f.getbbox(txt)
        return x1 - x0, y1 - y0
    return f.getsize(txt)

def highlight_tokens_pil(text, text_encoding, R_text, tokenizer,
                         show_zero_scores=False, min_score_to_show=0.1,
                         *, width_px=1200, font_size=24, font_path=None,
                         margin=6, score_font_scale=0.6):
    """Create PIL image with highlighted text tokens based on attention scores"""
    
    CLS_idx     = text_encoding.argmax(dim=-1)
    slice_      = R_text[CLS_idx, 1:CLS_idx]
    scores      = (slice_ / slice_.sum()).flatten()
    tokens      = [tokenizer.decode([i]) for i in tokenizer.encode(text)]
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min())

    entries = []
    for tok, sc in zip(tokens, scores_norm):
        t = tok.replace("</w>", " ").replace("Ġ", " ").strip()
        if not t: continue
        entries.append((t, None if (sc < min_score_to_show and not show_zero_scores) else float(sc)))

    def _load_font(size):
        if font_path:
            return ImageFont.truetype(font_path, size)
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()

    base_font   = _load_font(font_size)
    score_font  = _load_font(max(8, int(font_size*score_font_scale)))

    space_w, _  = _text_size(base_font, " ")
    token_h     = _text_size(base_font, "Hg")[1]
    score_up    = int(token_h * 0.40)
    line_h      = token_h + score_up + margin
    top_pad     = score_up

    lines, line, w_acc = [], [], 0
    for tok, sc in entries:
        s_txt      = f"{sc:.2f}" if sc is not None else ""
        w_tok, _   = _text_size(base_font, tok)
        w_score, _ = _text_size(score_font, s_txt)
        w_tot      = w_tok + w_score
        if w_acc + w_tot > width_px - margin*2:
            lines.append(line); line, w_acc = [], 0
        line.append((tok, sc, s_txt, w_tok, w_score))
        w_acc += w_tot + space_w
    if line: lines.append(line)

    img_h = line_h*len(lines) + margin + top_pad
    img   = Image.new("RGBA", (width_px, img_h), "white")
    draw  = ImageDraw.Draw(img)

    y = margin//2 + top_pad
    for line in lines:
        x = margin
        for tok, sc, s_txt, w_tok, w_score in line:
            w_block = w_tok + w_score
            if sc is not None:
                draw.rectangle([x,
                                y - score_up,        # cover superscript too
                                x + w_block,
                                y + token_h],
                               fill=(0, 255, 0, int(sc*255)))
            draw.text((x, y), tok, font=base_font, fill="black")
            if sc is not None:
                draw.text((x + w_tok, y - score_up),
                          s_txt, font=score_font, fill="black")
            x += w_block + space_w
        y += line_h

    return img

def setup_models_and_transforms():
    """Initialize models and image transforms"""
    print("Setting up models and transforms...")
    
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    dino_transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        MaybeToTensor(),
        normalize,
    ])

    model_clip, _, preprocess_clip = open_clip.create_model_and_transforms(CLIP_MODEL)
    transform_list = preprocess_clip.transforms
    preprocess_clip = T.Compose([T.Resize(size=(224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True)] 
                                + transform_list[2:])
    clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL)

    # Create wrapped models for attention capture
    model_clip = create_wrapped_clip(CLIP_MODEL, DEVICE)
    model_clip = model_clip.enable_attention_capture()

    model_dino = create_wrapped_dinov2(DINO_MODEL, DEVICE)
    model_dino.enable_attention_capture()
    
    return model_clip, model_dino, preprocess_clip, dino_transform, clip_tokenizer

def load_msae():
    """Load the Multi-Stream Sparse Autoencoder"""
    print("Loading MS-SAE...")
    
    # Infer modality dimensions
    d_streams = {
        'dino': 1024,  # DINOv2 CLS/pool dim
        'clip_img': 768,
        'clip_txt': 768,
    }

    SAE_CHECKPOINT = Path('final_results/msae_open_global_with_cross/msae_checkpoint.pth')
    with open('final_results/msae_open_global_with_cross/run_config.json', 'r') as f:
        config = json.load(f)
    
    msae = MSAE_Global(
        d_streams=d_streams,
        n_latents=config['args']['n_latents'],
        k=config['args']['k'],
    ).to(DEVICE)
    
    msae.load_state_dict(torch.load(SAE_CHECKPOINT, map_location=DEVICE, weights_only=False))
    msae.eval()
    
    return msae

def generate_heatmap_visualization(msae, idx, dataset, model_clip, model_dino, clip_tokenizer, 
                  preprocess_clip, dino_transform, caption=None, show_titles=True, 
                  global_msae=True, use_cross_modal=True):
    """Process a single image-caption pair and generate comprehensive visualization"""
    
    if caption is None:
        caption = dataset[idx]['captions']
    clip_sim_text = caption
    sparc_text = caption
    img_clip = preprocess_clip(dataset[idx]['image']).unsqueeze(0).to(DEVICE)
    img_dino = dino_transform(dataset[idx]['image']).unsqueeze(0).to(DEVICE)
    tokenized_sparc_text = clip_tokenizer(caption).to(DEVICE)
    k = None
    
    clip_txt_relevance, clip_img_relevance, dino_relevance = interpret_sparc(
        tokenized_sparc_text, model_clip, model_dino, img_clip, img_dino, msae, DEVICE,
        k, start_layer=-1, start_layer_text=-1, global_msae=global_msae, use_cross_modal=use_cross_modal
    )
    
    texts = clip_tokenizer(caption).to(DEVICE)
    clip_txt_sim_relevance, clip_img_sim_relevance = interpret_clip(model=model_clip, image=img_clip, texts=texts, 
                                                               device=DEVICE, start_layer=-1)
    
    gradcam_output = compute_gradcam(idx, k, msae, model_dino, model_clip, clip_tokenizer,
                                     sparc_text=caption, clip_sim_text=caption, dataset=dataset,
                                     dino_transform=dino_transform, preprocess_clip=preprocess_clip,
                                     device=DEVICE, thresh=0.1, is_global=global_msae, use_cross_modal=use_cross_modal)
    
    image, gradcam_dino_latent, gradcam_clip_latent, gradcam_clip_sim, tokens, scores = gradcam_output
    
    # Create heatmap overlays first to get dimensions
    def create_heatmap_overlay(img, relevance_tensor):
        dim = int(relevance_tensor.numel() ** 0.5)
        image_relevance = relevance_tensor.reshape(1, 1, dim, dim)
        image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
        image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
        
        # Normalize relevance values, avoiding division by zero
        rel_min = image_relevance.min()
        rel_max = image_relevance.max()
        if rel_max > rel_min:
            image_relevance = (image_relevance - rel_min) / (rel_max - rel_min)
        else:
            image_relevance = np.zeros_like(image_relevance)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * image_relevance), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)/255
        cam = cam / np.max(cam)
        vis = np.uint8(255 * cam)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        return vis
    
    original_resized = image.resize((224, 224))
    img_array = np.array(original_resized)
    dino_overlay = create_heatmap_overlay(img_array, dino_relevance[0])
    clip_overlay = create_heatmap_overlay(img_array, clip_img_relevance[0]) 
    clip_sim_overlay = create_heatmap_overlay(img_array, clip_img_sim_relevance[0])
    
    images = [
        original_resized, dino_overlay, clip_overlay, gradcam_dino_latent, 
        gradcam_clip_latent, clip_sim_overlay, gradcam_clip_sim
    ]
    
    # Calculate width for text attribution to match image row
    # 7 images * 224 pixels each = 1568 pixels total width
    text_width_px = 7 * 224
    
    # Create text attribution images using PIL with matching width
    clip_sim_img = highlight_tokens_pil(
        text=clip_sim_text,
        text_encoding=texts[0],
        R_text=clip_txt_sim_relevance[0],
        tokenizer=clip_tokenizer,
        show_zero_scores=False,
        min_score_to_show=0.1,
        width_px=text_width_px,
        font_size=24
    )
    
    sparc_img = highlight_tokens_pil(
        text=sparc_text,
        text_encoding=tokenized_sparc_text[0],
        R_text=clip_txt_relevance[0],
        tokenizer=clip_tokenizer,
        show_zero_scores=False,
        min_score_to_show=0.1,
        width_px=text_width_px,
        font_size=24
    )
    
    # Calculate figure size based on generated images
    image_row_height = 224  # pixels
    text_img_height_1 = sparc_img.height  # pixels (SPARC first)
    text_img_height_2 = clip_sim_img.height  # pixels (CLIP sim second)
    
    # Convert to inches (assuming 100 DPI)
    dpi = 100
    fig_width = text_width_px / dpi
    fig_height = (image_row_height + text_img_height_1 + text_img_height_2 + 160) / dpi  # +160 for margins/spacing
    
    # Create figure with calculated size
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Calculate height ratios based on actual image heights
    image_ratio = image_row_height
    text_ratio_1 = text_img_height_1  # SPARC
    text_ratio_2 = text_img_height_2  # CLIP sim
    
    gs = plt.GridSpec(3, 7, height_ratios=[image_ratio, text_ratio_1, text_ratio_2], hspace=0.2, wspace=0.01)
    
    titles = [
        f"Original (idx:{idx})", "Relevancy Map\nDINO (SPARC)", "Relevancy Map\nCLIP (SPARC)", 
        "GradCAM\nDINO (SPARC)", "GradCAM\nCLIP (SPARC)", "Relevancy Map\nCLIP Sim", "GradCAM\nCLIP Sim"
    ]
    
    # Plot images
    for i, (img, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        if show_titles:
            ax.set_title(title, fontsize=10, pad=3)
        ax.axis('off')
    
    # Plot SPARC text attribution (first)
    ax_text_sparc = fig.add_subplot(gs[1, :])
    ax_text_sparc.imshow(np.array(sparc_img))
    ax_text_sparc.set_title("SPARC Text Attribution", fontsize=12, fontweight='bold', pad=10)
    ax_text_sparc.axis('off')
    
    # Plot CLIP Similarity text attribution (second)  
    ax_text_sim = fig.add_subplot(gs[2, :])
    ax_text_sim.imshow(np.array(clip_sim_img))
    ax_text_sim.set_title("CLIP Similarity Text Attribution", fontsize=12, fontweight='bold', pad=10)
    ax_text_sim.axis('off')

    return fig

def main(open_images_dir):
    """Main function to run batch caption visualization"""
    print("Starting batch caption visualization...")
    
    # Setup components
    model_clip, model_dino, preprocess_clip, dino_transform, clip_tokenizer = setup_models_and_transforms()
    msae = load_msae()
    
    # Load dataset
    print("Loading dataset...")
    dataset = OpenImagesDataset(open_images_dir, 'test')
    
    # Create output directory
    output_dir = 'figures_all/all_captions'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")
    
    # Load annotations using pandas for efficient grouping
    print("Loading annotations and grouping by class...")
    img_ids = [i[0] for i in dataset.samples]
    df = pd.read_csv(os.path.join(open_images_dir, 'labels/test-annotations-human-imagelabels-boxable.csv'))
    df = df[df['Confidence'] == 1]
    df = df[df['ImageID'].isin(img_ids)]
    df['ClassName'] = df['LabelName'].map(dataset.label_to_class)
    
    # Create mapping from ImageID to dataset index
    img_id_to_idx = {img_id: idx for idx, (img_id, _) in enumerate(dataset.samples)}
    df['DatasetIdx'] = df['ImageID'].map(img_id_to_idx)
    
    # Group by class and select first 50 per class
    max_per_class = 40
    selected_indices_set = set()
    
    for class_name in df['ClassName'].unique():
        if pd.isna(class_name):  # Skip unmapped classes
            continue
        
        class_df = df[df['ClassName'] == class_name]
        # Get unique dataset indices for this class (sorted for consistent "first 50")
        unique_indices = sorted(class_df['DatasetIdx'].unique())
        # Take first 50 (or all if fewer than 50)
        selected_indices = unique_indices[:max_per_class]
        
        # Add to set to avoid duplicates
        selected_indices_set.update(selected_indices)
        
        # print(f"Class '{class_name}': {len(selected_indices)} samples (out of {len(unique_indices)} total)")
    
    # Convert to sorted list for consistent processing order
    selected_indices_list = sorted(list(selected_indices_set))
    total_samples = len(selected_indices_list)
    print(f"\nTotal unique samples to process: {total_samples}")
    print(f"Total classes found: {len(df['ClassName'].unique()) - df['ClassName'].isna().sum()}")
    
    # Process selected samples (no duplicates)
    total_generated = 0
    
    for i, idx in enumerate(tqdm(selected_indices_list, desc="Processing samples")):
        try:
            # Process the sample
            fig = generate_heatmap_visualization(
                msae=msae,
                idx=idx,
                dataset=dataset,
                model_clip=model_clip,
                model_dino=model_dino,
                clip_tokenizer=clip_tokenizer,
                preprocess_clip=preprocess_clip,
                dino_transform=dino_transform,
                caption=None,  # Use dataset caption
                show_titles=True,
                global_msae=True,
                use_cross_modal=True
            )
            
            # Save the figure with idx format (since images can belong to multiple classes)
            save_path = os.path.join(output_dir, f"caption_heatmap_idx{idx}.jpg")
            fig.savefig(save_path, bbox_inches='tight', dpi=200)
            plt.close(fig)
            total_generated += 1
            
            # Clean up memory periodically
            if i % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing idx {idx}: {e}")
            continue
    
    print(f"\nBatch caption visualization complete! Generated {total_generated} visualizations.")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    open_images_dir = '/home/ubuntu/Projects/OpenImages/'
    main(open_images_dir) 