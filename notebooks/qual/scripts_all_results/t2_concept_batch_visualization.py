#!/usr/bin/env python3
"""
Batch Concept Visualization Script
Generates visualizations for all concepts discovered by the MS-SAE across multiple samples.
Based on the t2_aligned_concept_visualization_open.ipynb notebook.
"""

import os
import torch
import torchvision.transforms as T
import numpy as np
import cv2
import json
import pickle
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import open_clip

# Import custom modules
from sparc.heatmaps.attention_relevance import interpret_sparc, interpret_clip
from sparc.heatmaps.gradcam import compute_gradcam
from sparc.heatmaps.clip import create_wrapped_clip
from sparc.heatmaps.dino import create_wrapped_dinov2
from sparc.model.model_global import MultiStreamSparseAutoencoder
from sparc.feature_extract.extract_open_images import OpenImagesDataset

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DINO_MODEL = 'dinov2_vitl14_reg'          
CLIP_MODEL = 'hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K'
DINO_TAG = 'dinov2_vitl14_reg'          
CLIP_TAG = 'CLIP-ViT-L-14-DataComp'
BATCH_SIZE = 32

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

class MaybeToTensor(T.ToTensor):
    def __call__(self, pic):
        return pic if isinstance(pic, torch.Tensor) else super().__call__(pic)

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
    
    msae = MultiStreamSparseAutoencoder(
        d_streams=d_streams,
        n_latents=config['args']['n_latents'],
        k=config['args']['k'],
    ).to(DEVICE)
    
    msae.load_state_dict(torch.load(SAE_CHECKPOINT, map_location=DEVICE, weights_only=False))
    msae.eval()
    
    return msae

def load_concept_names():
    """Load or compute concept names by latent"""
    concept_file = 'final_results/msae_open_global_with_cross/concept_names_by_latent.pkl'
    
    if os.path.isfile(concept_file):
        print("Loading existing concept names...")
        with open(concept_file, 'rb') as f:
            concept_names_by_latent = pickle.load(f)
    else:
        print("Concept names file not found. Please run the concept discovery first.")
        raise FileNotFoundError(f"Could not find {concept_file}")
    
    return concept_names_by_latent

def get_concept_latents_and_samples(concept, concept_names_by_latent, dataset, verbose=True):
    """Get latent indices and sample indices for a given concept"""
    target_k = []

    for i in range(len(concept_names_by_latent)):
        had_concept = False
        for stream in ['dino', 'clip_img', 'clip_txt']:
            if concept_names_by_latent[i][stream] == concept:
                had_concept = True
        if had_concept:
            target_k.append(i)

    all_img_ids = [i[0] for i in dataset.samples]

    img_ids = []
    for img_id, labels in dataset.image_to_labels.items():
        if concept in labels:
            img_ids.append(img_id)
    indices = [all_img_ids.index(i) for i in img_ids]
    
    if verbose:
        print(f"Concept '{concept}': {len(target_k)} latents, {len(indices)} samples")

    return target_k, indices

def visualize_concept_with_text(concept, idx, target_k, dataset, models, transforms, msae):
    """Create visualization for a single concept and sample"""
    model_clip, model_dino, clip_tokenizer = models
    preprocess_clip, dino_transform = transforms
    
    def indefinite_article(word):
        return 'an' if word[0].lower() in 'aeiou' else 'a'
    
    clip_sim_text = f"{indefinite_article(concept)} {concept}"
    sparc_text = dataset[idx]['captions']
    
    img_clip = preprocess_clip(dataset[idx]['image']).unsqueeze(0).to(DEVICE)
    img_dino = dino_transform(dataset[idx]['image']).unsqueeze(0).to(DEVICE)
    tokenized_sparc_texts = clip_tokenizer(sparc_text).to(DEVICE)
    
    start_layer_image = -1
    start_layer_text = -1
    
    clip_txt_relevance, clip_img_relevance, dino_relevance = interpret_sparc(
        tokenized_sparc_texts, model_clip, model_dino, img_clip, img_dino, msae, DEVICE,
        target_k, start_layer=start_layer_image, start_layer_text=start_layer_text, 
        global_msae=True, use_cross_modal=False
    )
    
    tokenized_clip_texts = clip_tokenizer(clip_sim_text).to(DEVICE)
    clip_txt_sim_relevance, clip_img_sim_relevance = interpret_clip(
        model=model_clip, image=img_clip, texts=tokenized_clip_texts, device=DEVICE, 
        start_layer=start_layer_image, start_layer_text=start_layer_text
    )
    
    gradcam_output = compute_gradcam(idx, target_k, msae, model_dino, model_clip, clip_tokenizer,
                                     sparc_text=sparc_text, clip_sim_text=clip_sim_text, dataset=dataset,
                                     dino_transform=dino_transform, preprocess_clip=preprocess_clip,
                                     device=DEVICE, thresh=0.1, is_global=True, use_cross_modal=False)
    
    image, gradcam_dino_latent, gradcam_clip_latent, gradcam_clip_sim, tokens, scores = gradcam_output
    
    # Text attribution processing
    CLS_idx = tokenized_sparc_texts[0].argmax(dim=-1)
    R_text = clip_txt_relevance[0][CLS_idx, 1:CLS_idx]
    text_scores = R_text / R_text.sum()
    text_scores = text_scores.flatten()
    text_tokens_decoded = [clip_tokenizer.decode([a]) for a in clip_tokenizer.encode(sparc_text)]
    scores_norm = (text_scores - text_scores.min()) / (text_scores.max() - text_scores.min())
    
    # Create visualization
    fig = plt.figure(figsize=(18, 5))
    gs = plt.GridSpec(3, 7, height_ratios=[4, 2, 1], hspace=0.3, wspace=0.01)
    
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
            # All values are the same, set to middle value (0.5) or zero
            image_relevance = np.zeros_like(image_relevance)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * image_relevance), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)/255
        cam = cam / np.max(cam)
        vis = np.uint8(255 * cam)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        return vis
    
    original_resized = image.resize((224, 224))
    img = np.array(original_resized)
    dino_overlay = create_heatmap_overlay(img, dino_relevance[0])
    clip_overlay = create_heatmap_overlay(img, clip_img_relevance[0]) 
    r_image_overlay = create_heatmap_overlay(img, clip_img_sim_relevance[0])
    
    images = [
        original_resized, dino_overlay, clip_overlay, gradcam_dino_latent, 
        gradcam_clip_latent, r_image_overlay, gradcam_clip_sim
    ]
    
    titles = [
        "Original", "Relevancy Map\nDINO (SPARC)", "Relevancy Map\nCLIP (SPARC)", 
        "GradCAM\nDINO (SPARC)", "GradCAM\nCLIP (SPARC)", "Relevancy Map\nCLIP Sim", "GradCAM\nCLIP Sim"
    ]
    
    for i, (img_vis, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img_vis)
        ax.set_title(title, fontsize=10, pad=3)
        ax.axis('off')
    
    # Text attribution visualization
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)
    ax_text.axis('off')

    ax_text.text(0.02, 0.8, "SPARC Text Attribution", fontsize=12, fontweight='bold', ha='left', va='bottom')

    x_pos = 0.02
    y_pos = 0.5

    for token, score in zip(text_tokens_decoded, scores_norm):
        clean_token = token.replace('</w>', ' ').replace('Ġ', ' ').strip()
        if clean_token:
            intensity = float(score)

            if intensity > 0.0:
                green_intensity = intensity * 0.8
                green_color = (1 - green_intensity, 1.0, 1 - green_intensity)
                ax_text.text(x_pos, y_pos, clean_token, fontsize=12, 
                           bbox=dict(boxstyle="round,pad=0.1", facecolor=green_color, alpha=1.0),
                           ha='left', va='center')
            else:
                ax_text.text(x_pos, y_pos, clean_token, fontsize=12, ha='left', va='center')
            
            ax_text.text(x_pos, y_pos + 0.2, f"{intensity:.2f}", fontsize=8, ha='left', va='center')

            x_pos += len(clean_token) * 0.012 + 0.015

            if x_pos > 0.95:
                x_pos = 0.02
                y_pos -= 0.4

    # Info section
    info_ax = fig.add_subplot(gs[2, :])
    info_ax.text(0.5, 0.5, f"Concept: {concept} | Sample Index: {idx}", 
                ha='center', va='center', fontsize=10)
    info_ax.axis('off')
    
    return fig

def main(open_images_dir):
    """Main function to run batch concept visualization"""
    print("Starting batch concept visualization...")
    
    # Setup components
    model_clip, model_dino, preprocess_clip, dino_transform, clip_tokenizer = setup_models_and_transforms()
    msae = load_msae()
    concept_names_by_latent = load_concept_names()
    
    # Load dataset
    print("Loading dataset...")
    dataset = OpenImagesDataset(open_images_dir, 'test')
    
    # Find concepts consistent across all modalities
    print("Finding consistent concepts...")
    final_concepts = set(i['dino'] for i in concept_names_by_latent) \
        .intersection(i['clip_img'] for i in concept_names_by_latent) \
        .intersection(i['clip_txt'] for i in concept_names_by_latent)
    
    # Remove 'None' if it exists
    final_concepts.discard('None')
    
    print(f"Found {len(final_concepts)} consistent concepts: {sorted(list(final_concepts))}")
    
    # Create output directory
    output_dir = 'figures_all/all_concepts'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")
    
    models = (model_clip, model_dino, clip_tokenizer)
    transforms = (preprocess_clip, dino_transform)
    
    # Process each concept
    total_generated = 0
    for concept in tqdm(final_concepts, desc="Processing concepts"):
        try:
            target_k, indices = get_concept_latents_and_samples(
                concept, concept_names_by_latent, dataset, verbose=False
            )
            
            if not indices:
                print(f"No samples found for concept: {concept}")
                continue
                
            # Process up to 50 samples per concept
            for concept_idx in range(min(50, len(indices))):
                idx = indices[concept_idx]
                
                try:
                    fig = visualize_concept_with_text(
                        concept, idx, target_k, dataset, models, transforms, msae
                    )

                    save_path = os.path.join(output_dir, f"{concept}_idx{idx}.jpg")
                    fig.savefig(save_path, bbox_inches='tight', dpi=200)
                    plt.close(fig)
                    total_generated += 1
                    
                except Exception as e:
                    print(f"Error processing {concept} idx {idx}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error processing concept {concept}: {e}")
            continue
    
    print(f"\nBatch visualization complete! Generated {total_generated} visualizations.")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    open_images_dir = '/home/ubuntu/Projects/OpenImages/'
    main(open_images_dir) 