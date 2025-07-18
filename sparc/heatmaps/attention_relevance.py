"""
Attention rollout methods for transformer models.

This module provides attention rollout implementations for CLIP, DINO,
and sparse autoencoder models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Union


def get_attention_blocks(model: nn.Module, model_type: str) -> Tuple[List[nn.Module], Optional[List[nn.Module]]]:
    """Get attention blocks from different model types."""
    if model_type == 'clip':
        image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
        text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())
        return image_attn_blocks, text_attn_blocks
    else:
        image_attn_blocks = list(dict(model.blocks.named_children()).values())
        return image_attn_blocks, None


def compute_attention_relevancy(
    target: torch.Tensor, 
    attn_blocks: List[nn.Module], 
    device: str, 
    batch_size: int, 
    start_layer: int
) -> torch.Tensor:
    """Compute attention rollout relevancy."""
    num_tokens = attn_blocks[-1].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=attn_blocks[-1].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    
    for i, blk in enumerate(attn_blocks):
        if i < start_layer:
            continue
        grad = torch.autograd.grad(target, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    return R


def get_all_latents(
    texts: torch.Tensor, 
    msae: nn.Module, 
    dino_model: nn.Module, 
    dino_image: torch.Tensor, 
    clip_model: nn.Module, 
    clip_image: torch.Tensor, 
    global_msae: bool,
    return_original_features: bool = False
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
]:
    """Get latent representations from all modalities.
    Optionally returns original features as well."""
    batch_size = texts.shape[0]
    
    if global_msae:
        clip_image_features = clip_model.encode_image(clip_image)
        clip_image_features = clip_image_features / clip_image_features.norm(dim=-1, keepdim=True)
        clip_text_features = clip_model.encode_text(texts)
        clip_text_features = clip_text_features / clip_text_features.norm(dim=-1, keepdim=True)
        dino_image_features = dino_model(dino_image)
        dino_image_features = dino_image_features / dino_image_features.norm(dim=-1, keepdim=True)
        
        msae_input = {
            'dino': dino_image_features, 
            'clip_img': clip_image_features, 
            'clip_txt': clip_text_features
        }
        msae_output = msae(msae_input)
        clip_txt_latent = msae_output['sparse_codes_clip_txt']
        clip_img_latent = msae_output['sparse_codes_clip_img']
        dino_latent = msae_output['sparse_codes_dino']
    else:
        clip_text_features = clip_model.encode_text(texts)
        clip_text_features = clip_text_features / clip_text_features.norm(dim=-1, keepdim=True)
        clip_txt_latent = msae({'clip_txt': clip_text_features})['sparse_codes_clip_txt']
        
        clip_image_features = clip_model.encode_image(clip_image)
        clip_image_features = clip_image_features / clip_image_features.norm(dim=-1, keepdim=True)
        clip_img_latent = msae({'clip_img': clip_image_features})['sparse_codes_clip_img']
        
        dino_image_features = dino_model(dino_image)
        dino_image_features = dino_image_features / dino_image_features.norm(dim=-1, keepdim=True)
        dino_latent = msae({'dino': dino_image_features})['sparse_codes_dino']
    
    if return_original_features:
        return clip_txt_latent, clip_img_latent, dino_latent, clip_text_features, clip_image_features, dino_image_features, batch_size
    else:
        return clip_txt_latent, clip_img_latent, dino_latent, batch_size


def interpret_clip(
    image: torch.Tensor, 
    texts: torch.Tensor, 
    model: nn.Module, 
    device: str, 
    start_layer: int = -1, 
    start_layer_text: int = -1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Interpret CLIP model using attention rollout.
    
    Args:
        image: Input image tensor
        texts: Input text tokens
        model: CLIP model
        device: Device to use
        start_layer: Starting layer for image attention rollout
        start_layer_text: Starting layer for text attention rollout
        
    Returns:
        Tuple of (text_relevance, image_relevance)
    """
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    
    # Forward
    image_features = model.encode_image(images)
    text_features = model.encode_text(texts)

    # Normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    if start_layer == -1:
        start_layer = len(image_attn_blocks) - 1
    
    image_relevance = compute_attention_relevancy(
        one_hot, image_attn_blocks, device, batch_size, start_layer
    )
    image_relevance = image_relevance[:, 0, 1:]

    text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

    if start_layer_text == -1:
        start_layer_text = len(text_attn_blocks) - 1

    text_relevance = compute_attention_relevancy(
        one_hot, text_attn_blocks, device, batch_size, start_layer_text
    )
   
    return text_relevance, image_relevance


def interpret_sparc(
    texts: torch.Tensor, 
    clip_model: nn.Module, 
    dino_model: nn.Module, 
    clip_image: torch.Tensor, 
    dino_image: torch.Tensor, 
    msae: nn.Module, 
    device: str, 
    k: int, 
    start_layer: int, 
    start_layer_text: int, 
    global_msae: bool,
    use_cross_modal = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Interpret SPARC using attention rollout.
    
    Args:
        texts: Input text tokens
        clip_model: CLIP model
        dino_model: DINO model
        clip_image: CLIP preprocessed image
        dino_image: DINO preprocessed image
        msae: Multi-modal sparse autoencoder
        device: Device to use
        k: Feature index
        start_layer: Starting layer for image attention rollout
        start_layer_text: Starting layer for text attention rollout
        global_msae: Whether to use global MSAE mode
        use_cross_modal: Whether to use cross-modal sparc latents
        
    Returns:
        Tuple of (clip_txt_relevance, clip_img_relevance, dino_relevance)
    """
    clip_txt_latent, clip_img_latent, dino_latent, batch_size = get_all_latents(
        texts, msae, dino_model, dino_image, clip_model, clip_image, global_msae
    )
    if use_cross_modal:
        assert k is None, "k must be None when using cross-modal sparc latents"
        clip_txt_target = clip_txt_latent @ clip_img_latent.T
        clip_img_target = clip_img_latent @ clip_txt_latent.T
        dino_target = dino_latent @ clip_txt_latent.T
    else:
        if k is None or (hasattr(k, '__len__') and len(k) == 0):
            clip_txt_target = torch.sum(clip_txt_latent)
            clip_img_target = torch.sum(clip_img_latent)
            dino_target = torch.sum(dino_latent)
        else:
            clip_txt_target = torch.sum(clip_txt_latent[:, k])
            clip_img_target = torch.sum(clip_img_latent[:, k])
            dino_target = torch.sum(dino_latent[:, k])
    
    clip_model.zero_grad()
    dino_model.zero_grad()
    
    # Get attention blocks for both models
    clip_img_attn_blocks, clip_txt_attn_blocks = get_attention_blocks(clip_model, 'clip')
    dino_img_attn_blocks, _ = get_attention_blocks(dino_model, 'dino')
    
    # Compute relevancy using respective models
    clip_txt_relevance = compute_attention_relevancy(
        clip_txt_target, clip_txt_attn_blocks, device, batch_size, start_layer_text
    )
    clip_img_relevance = compute_attention_relevancy(
        clip_img_target, clip_img_attn_blocks, device, batch_size, start_layer
    )
    dino_relevance = compute_attention_relevancy(
        dino_target, dino_img_attn_blocks, device, batch_size, start_layer
    )
    
    clip_img_relevance = clip_img_relevance[:, 0, 1:]
    dino_relevance = dino_relevance[:, 0, 5:]  # Skip CLS + register tokens
   
    return clip_txt_relevance, clip_img_relevance, dino_relevance 