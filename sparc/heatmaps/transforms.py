"""
Transform functions for different model types (CLIP, DINO, etc.).

This module contains reshape and preprocessing functions needed for 
GradCAM and other visualization methods with different model architectures.
"""

import torch
from typing import Any


def reshape_transform_clip(tensor: torch.Tensor, height: int = 16, width: int = 16) -> torch.Tensor:
    """
    Reshape CLIP vision transformer tokens for GradCAM.
    
    Args:
        tensor: Input tensor from CLIP vision transformer
        height: Height of the feature map grid
        width: Width of the feature map grid
        
    Returns:
        Reshaped tensor in CNN format [B, C, H, W]
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # Bring channels to first dimension, like in CNNs
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform_dino(tokens: torch.Tensor) -> torch.Tensor:
    """
    Convert ViT/DINO token sequence to CNN-style feature map for Grad-CAM.

    Args:
        tokens: Shape [B, 1 + n_reg + n_patch, C] (CLS, registers, patches)

    Returns:
        Shape [B, C, H, W] containing only the patch tokens laid out as H×W grid
    """
    # Number of register tokens used by the backbone (4 for DINO-v2 B/L; 0 for CLIP)
    n_reg = 4

    # Slice out patch tokens: skip CLS (idx 0) + registers (idx 1..n_reg)
    patch_tok = tokens[:, 1 + n_reg:, :]  # [B, N_patch, C]

    B, N, C = patch_tok.shape
    H = W = int(N ** 0.5)  # 256 → 16×16
    assert H * W == N, f"Patch count {N} is not a square number"

    patch_map = patch_tok.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
    return patch_map


def reshape_transform_text(tokens: torch.Tensor) -> torch.Tensor:
    """
    Reshape text tokens for GradCAM compatibility.
    
    Args:
        tokens: [B, L, C] (batch, sequence length, channels)
        
    Returns:
        Reshaped tensor [B, C, 1, L]
    """
    B, L, C = tokens.shape
    return tokens.permute(0, 2, 1).unsqueeze(2)