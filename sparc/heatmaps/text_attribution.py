"""
Text attribution methods for multimodal models.

This module provides gradient-based text attribution methods that work
with CLIP and sparse autoencoder models.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Any, Tuple

from .tokenization import ids_to_tokens, tokens_to_words


def gradtext_lastlayer(
    model_clip: nn.Module,
    msae: Optional[nn.Module],
    token_ids: torch.Tensor,
    target_layer: nn.Module,
    k: int,
    *,
    device: str = "cuda",
    word_level: bool = True,
    is_global: bool = False,
    clip_img_feat: Optional[torch.Tensor] = None,
    dino_feat: Optional[torch.Tensor] = None,
    tokenizer: Any,
    use_cross_modal: bool = True
) -> Tuple[List[str], torch.Tensor]:
    """
    Compute gradient-based text attribution for the last layer.
    
    Args:
        model_clip: CLIP model
        msae: Multi-modal sparse autoencoder (optional)
        token_ids: Input token IDs
        target_layer: Layer to compute gradients for
        k: Feature index for SAE
        device: Device to use
        word_level: Whether to return word-level or token-level attributions
        is_global: Whether to use global SAE mode
        clip_img_feat: CLIP image features (required for global mode)
        dino_feat: DINO features (required for global mode)
        tokenizer: Tokenizer for converting IDs to tokens
        use_cross_modal: Whether to use cross-modal features
        
    Returns:
        Tuple of (tokens/words, scores)
    """
    device = torch.device(device)
    token_ids = token_ids.to(device)
    model_clip = model_clip.to(device).eval()

    saved = {}
    
    def fwd_hook(_m, _inp, out):
        saved["A"] = out  # keep the original tensor
        out.retain_grad()  # tell PyTorch to store its grad
        
    h = target_layer.register_forward_hook(fwd_hook)

    model_clip.zero_grad(set_to_none=True)
    if msae is not None:
        msae.zero_grad(set_to_none=True)

    feat_vec = model_clip.encode_text(token_ids)

    if msae is not None:
        if is_global:
            assert clip_img_feat is not None and dino_feat is not None
            out = msae({
                'dino': dino_feat,
                'clip_img': clip_img_feat,
                'clip_txt': feat_vec
            })
        else:
            out = msae({'clip_txt': feat_vec})
        
        if use_cross_modal:
            assert is_global, "Global mode must be enabled for cross-modal computation"
            # Cross-modal computation: text features similarity with image features
            clip_txt_latent = out['sparse_codes_clip_txt']
            clip_img_latent = out['sparse_codes_clip_img']
            obj = (clip_txt_latent @ clip_img_latent.T).sum()
        else:
            obj = feat_vec.norm(p=2, dim=1).sum()
    else:
        obj = feat_vec.norm(p=2, dim=1).sum()  # any scalar you like

    obj.backward()

    A = saved["A"]  # [B, L, C]
    dLdA = A.grad  # [B, L, C] – now it's not None!

    alpha = dLdA.mean(dim=1, keepdim=True)  # [B, 1, C]
    cam = torch.relu((alpha * A).sum(dim=-1))  # [B, L]
    cam = cam / (cam.max(dim=1, keepdim=True).values + 1e-8)
    cam = cam.squeeze(0).detach().cpu()  # [L]

    h.remove()

    sub_tokens = ids_to_tokens(tokenizer, token_ids.squeeze(0).tolist())
    if word_level:
        tokens, scores = tokens_to_words(sub_tokens, cam)
    else:
        tokens, scores = sub_tokens, cam

    return tokens, scores


def gradtext(
    msae: nn.Module,
    token_ids: torch.Tensor,
    model: nn.Module,
    k: List[int],
    tokenizer: Any,
    device: str = "cuda",
    embedding_layer: Optional[nn.Module] = None,
    norm_order: int = 2,
    word_level: bool = True,
    is_global: bool = True,
    clip_img_feat: Optional[torch.Tensor] = None,
    dino_feat: Optional[torch.Tensor] = None
) -> Tuple[List[str], torch.Tensor]:
    """
    Compute gradient-based text attribution using embedding layer gradients.
    
    Args:
        msae: Multi-modal sparse autoencoder
        token_ids: Input token IDs
        model: Text model
        k: Feature indices for SAE
        tokenizer: Tokenizer for converting IDs to tokens
        device: Device to use
        embedding_layer: Embedding layer to hook (auto-detected if None)
        norm_order: Norm order for objective
        word_level: Whether to return word-level or token-level attributions
        is_global: Whether to use global SAE mode
        clip_img_feat: CLIP image features (required for global mode)
        dino_feat: DINO features (required for global mode)
        
    Returns:
        Tuple of (tokens/words, scores)
    """
    if is_global:
        assert clip_img_feat is not None and dino_feat is not None, \
            "clip_img_feat and dino_feat must be provided for global gradtext"
            
    device = torch.device(device)
    token_ids = token_ids.to(device)
    model = model.to(device).eval()

    # Locate embedding layer
    if embedding_layer is None:
        embedding_layer = next(m for m in model.modules() if isinstance(m, nn.Embedding))

    saved = {}

    def hook(_m, _inp, out):
        saved["emb"] = out
        out.retain_grad()

    h = embedding_layer.register_forward_hook(hook)

    # Forward with grad
    model.zero_grad(set_to_none=True)
    feat_vec = (model.encode_text(token_ids)
                if hasattr(model, "encode_text") else model(token_ids))
    
    if is_global:
        output = msae({'dino': dino_feat, 'clip_img': clip_img_feat, 'clip_txt': feat_vec})
    else:
        output = msae({'clip_txt': feat_vec})
    
    # Scalar objective
    sparse_codes = output['sparse_codes_clip_txt']
    max_dim = sparse_codes.argmax(dim=1)
    obj = sparse_codes[:, max_dim:max_dim+1]

    # Backward
    obj.backward()
    grads = saved["emb"].grad.norm(dim=-1).squeeze(0)  # (seq_len,)

    # Normalize 0-1
    grads = (grads - grads.min()) / (grads.max() - grads.min() + 1e-12)

    h.remove()

    # IDs → sub-tokens → (optionally) words
    sub_tokens = ids_to_tokens(tokenizer, token_ids.squeeze(0).tolist())

    if word_level:
        tokens, scores = tokens_to_words(sub_tokens, grads)
    else:
        tokens, scores = sub_tokens, grads

    return tokens, scores.cpu() 