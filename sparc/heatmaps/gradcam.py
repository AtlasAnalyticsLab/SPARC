"""
GradCAM visualization functions for multimodal models.

This module provides GradCAM implementations that work with CLIP, DINO,
and sparse autoencoder models.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Optional, Any, Tuple

from pytorch_grad_cam import (
    GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, 
    XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

from .transforms import reshape_transform_dino, reshape_transform_clip
from .models import TextGuidedSAE, TextGuidedClip
from .text_attribution import gradtext_lastlayer


def visualize_cam(
    image: Any,  # PIL Image
    preprocess_img_func: Any,
    model: nn.Module,
    target_layers: List[nn.Module],
    method_name: str = 'gradcam',
    labels: Optional[List[str]] = None,
    aug_smooth: bool = False,
    eigen_smooth: bool = False,
    thresh: Optional[float] = None,
    is_dino: bool = True,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Generate a CAM visualization for the given image using the specified method.
    
    Args:
        image: PIL Image only
        preprocess_img_func: Function to preprocess the image
        model: The model to use
        target_layers: List of target layers to use for CAM
        method_name: CAM method (gradcam, scorecam, etc.)
        labels: List of class labels
        aug_smooth: Apply test time augmentation
        eigen_smooth: Apply eigen smoothing
        thresh: Threshold for clipping CAM values
        is_dino: Whether using DINO model (affects reshape transform)
        device: Device to use for computation
        
    Returns:
        The visualization with heatmap overlay
    """
    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad
    }
    
    if method_name not in methods:
        raise ValueError(f"Method {method_name} not supported. Choose from: {list(methods.keys())}")

    # Prepare image for CAM input tensor
    input_tensor = preprocess_img_func(image).unsqueeze(0).to(device)

    # Prepare image for visualization overlay
    rgb_img = np.array(image.convert('RGB')).astype(np.float32) / 255.0
    
    # Resize for consistent overlay size
    try:
        target_size = (input_tensor.shape[-1], input_tensor.shape[-2])  # W, H for cv2.resize
    except IndexError:
        print("Warning: Could not determine target size from input_tensor shape. Defaulting to (224, 224).")
        target_size = (224, 224)
    rgb_img = cv2.resize(rgb_img, target_size)

    # Initialize CAM method
    reshape_func = reshape_transform_dino if is_dino else reshape_transform_clip
    
    if method_name == "ablationcam":
        cam = methods[method_name](
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_func,
            ablation_layer=AblationLayerVit()
        )
    else:
        cam = methods[method_name](
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_func
        )
    
    # Set batch size for methods that support batching
    cam.batch_size = 32
    
    # Generate the CAM
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=None,  # If None, returns the map for the highest scoring category
        eigen_smooth=eigen_smooth,
        aug_smooth=aug_smooth
    )
    
    # Get the first image in the batch
    grayscale_cam = grayscale_cam[0, :]
    
    if thresh is not None:
        grayscale_cam = np.clip(
            (grayscale_cam - thresh) / (1.0 - thresh + 1e-7), 0, 1
        )

    # Create visualization using the prepared rgb_img
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return cam_image


def compute_gradcam(
    i: int, 
    k: int, 
    msae: nn.Module, 
    model_dino: nn.Module, 
    model_clip: nn.Module, 
    clip_tokenizer: Any, 
    sparc_text: str, 
    clip_sim_text: List[str],
    dataset: Dataset, 
    dino_transform: Any, 
    preprocess_clip: Any, 
    device: str,
    thresh: Optional[float] = None, 
    is_global: bool = True,
    use_cross_modal: bool = True
) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, List[str], torch.Tensor]:
    """
    Compute GradCAM visualizations for a given image and concept.
    
    Args:
        i: Dataset index
        k: Feature index for SAE
        msae: Multi-stream sparse autoencoder
        model_dino: DINO model
        model_clip: CLIP model
        clip_tokenizer: CLIP tokenizer
        sparc_text: Text for SPARC analysis
        clip_sim_text: Text for CLIP similarity
        dataset: Image dataset
        dino_transform: DINO preprocessing transform
        preprocess_clip: CLIP preprocessing transform
        device: Device to use
        thresh: Threshold for CAM visualization
        is_global: Whether to use global MSAE mode
        use_cross_modal: Whether to use cross-modal features
        
    Returns:
        Tuple of (image, dino_cam, clip_cam, clip_sim_cam, tokens, scores)
    """
    
    image = dataset[i]['image']

    # Create model wrappers for GradCAM
    gradcam_dino_latent = TextGuidedSAE(
        msae, k, model_dino, model_clip, clip_tokenizer, 
        device, sparc_text, stream='dino', is_global=is_global, use_cross_modal=use_cross_modal
    )
    gradcam_clip_latent = TextGuidedSAE(
        msae, k, model_dino, model_clip, clip_tokenizer, 
        device, sparc_text, stream='clip_img', is_global=is_global, use_cross_modal=use_cross_modal
    )
    
    # Define target layers
    target_layers_clip = [model_clip.visual.transformer.resblocks[-1].ln_1]
    target_layers_dino = [model_dino.blocks[-1].norm1]

    # Generate CAM visualizations
    cam_image_dino_latent = visualize_cam(
        image=image,
        preprocess_img_func=dino_transform,
        model=gradcam_dino_latent,
        target_layers=target_layers_dino,
        method_name="gradcam", 
        thresh=thresh,
        is_dino=True
    )

    cam_image_clip_latent = visualize_cam(
        image=image,
        preprocess_img_func=preprocess_clip,
        model=gradcam_clip_latent,
        target_layers=target_layers_clip,
        method_name="gradcam", 
        thresh=thresh,
        is_dino=False
    )
    
    # CLIP similarity baseline
    gradcam_clip = TextGuidedClip(model_clip, clip_tokenizer, device, clip_sim_text)
    cam_image_clip = visualize_cam(
        image=image,
        preprocess_img_func=preprocess_clip,
        model=gradcam_clip,
        target_layers=target_layers_clip,
        method_name="gradcam", 
        thresh=thresh,
        is_dino=False
    )

    # Compute features for text attribution
    dino_features = model_dino(dino_transform(image).unsqueeze(0).to(device))
    clip_img_features = model_clip.encode_image(preprocess_clip(image).unsqueeze(0).to(device))
    dino_features = dino_features / dino_features.norm(dim=1, keepdim=True)
    clip_img_features = clip_img_features / clip_img_features.norm(dim=1, keepdim=True)

    # Text attribution
    token_ids = clip_tokenizer([sparc_text]).to(device)   
    target_layer = model_clip.transformer.resblocks[-1].ln_1

    tokens, scores = gradtext_lastlayer(
        model_clip,
        msae,
        token_ids,
        target_layer,
        k,
        device=device,
        word_level=True,
        is_global=is_global,
        clip_img_feat=clip_img_features,
        dino_feat=dino_features,
        tokenizer=clip_tokenizer,
        use_cross_modal=use_cross_modal,
    )
    
    return image, cam_image_dino_latent, cam_image_clip_latent, cam_image_clip, tokens, scores 