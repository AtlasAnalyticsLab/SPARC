"""
Model wrapper classes and dataset utilities.

This module contains wrapper classes for different models (CLIP, DINO, SAE)
that enable them to work with GradCAM and other visualization methods.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Optional, Any, Dict


class CocoImagesDataset(Dataset):
    """Dataset wrapper for COCO images with captions."""
    
    def __init__(self, coco_ds: Dataset, transform: Optional[Any] = None):
        self.coco_ds = coco_ds
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.coco_ds)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image, caps = self.coco_ds[idx]
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'captions': ' '.join(caps), 'idx': idx}


class TextGuidedSAE(nn.Module):
    """Wrapper for using SAE with text guidance for GradCAM."""
    
    def __init__(
        self, 
        msae: nn.Module, 
        k: int, 
        model_dino: nn.Module, 
        model_clip: nn.Module, 
        clip_tokenizer: Any, 
        device: str, 
        text: str,
        stream: str, 
        is_global: bool = False,
        use_cross_modal: bool = True
    ):
        super().__init__()
        self.msae = msae
        self.model_dino = model_dino
        self.clip = model_clip
        self.tokenizer = clip_tokenizer
        self.text = text
        self.device = device
        self.stream = stream
        self.k = k
        self.is_global = is_global
        self.use_cross_modal = use_cross_modal
        assert self.stream in ['dino', 'clip_img'], "Stream must be either 'dino' or 'clip_img'"
        if self.use_cross_modal:
            assert self.is_global, "Global mode must be enabled for cross-modal computation"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        text_tokens = self.tokenizer(self.text).to(self.device)
        
        dino_features = self.model_dino(x)
        clip_img_features = self.clip.encode_image(x)
        clip_txt_features = self.clip.encode_text(text_tokens)
            
        # Normalize features
        dino_features = dino_features / dino_features.norm(dim=1, keepdim=True)
        clip_img_features = clip_img_features / clip_img_features.norm(dim=1, keepdim=True)
        clip_txt_features = clip_txt_features / clip_txt_features.norm(dim=1, keepdim=True)
        
        if self.is_global:
            output = self.msae.forward({
                'dino': dino_features, 
                'clip_img': clip_img_features,
                'clip_txt': clip_txt_features
            })
        elif self.stream == 'dino':
            output = self.msae.forward({'dino': dino_features})
        elif self.stream == 'clip_img':
            output = self.msae.forward({'clip_img': clip_img_features})
        elif self.stream == 'clip_txt':
            output = self.msae.forward({'clip_txt': clip_txt_features})
                    
        if self.use_cross_modal:
            assert self.is_global, "Global mode must be enabled for cross-modal computation"
            # Get sparse codes for cross-modal computation
            clip_txt_latent = output['sparse_codes_clip_txt'] if 'sparse_codes_clip_txt' in output else None
            clip_img_latent = output['sparse_codes_clip_img'] if 'sparse_codes_clip_img' in output else None
            dino_latent = output['sparse_codes_dino'] if 'sparse_codes_dino' in output else None
            
            # Compute cross-modal similarities like in interpret_sparc
            if self.stream == 'dino' and dino_latent is not None and clip_txt_latent is not None:
                logits = dino_latent @ clip_txt_latent.T
            elif self.stream == 'clip_img' and clip_img_latent is not None and clip_txt_latent is not None:
                logits = clip_img_latent @ clip_txt_latent.T

        else:
            # Compute similarity (original logic)
            if len(self.k)>0:
                logits = output[f'sparse_codes_{self.stream}'][:, self.k]
                max_dim = logits.argmax()
                logits = logits[:, max_dim:max_dim+1]

            else:
                logits = output[f'sparse_codes_{self.stream}'].sum(dim=1, keepdim=True)
            
            #         logits = logits.mean(axis=1, keepdim=True)
            #         logits = logits.norm(dim=1, p=2, keepdim=True)
            
            #         print(logits.detach().cpu().numpy(), max_dim.item())

        return logits


class TextGuidedClip(nn.Module):
    """Wrapper for CLIP with text guidance for GradCAM."""
    
    def __init__(self, model_clip: nn.Module, clip_tokenizer: Any, device: str, labels: List[str]):
        super().__init__()
        self.clip = model_clip
        self.tokenizer = clip_tokenizer
        self.labels = labels
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        text_tokens = self.tokenizer(self.labels).to(self.device)
        
        image_features = self.clip.encode_image(x)
        text_features = self.clip.encode_text(text_tokens)
            
        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        # Compute similarity
        logits = 100 * torch.matmul(image_features, text_features.T)
        return logits 