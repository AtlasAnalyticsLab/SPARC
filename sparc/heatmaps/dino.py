import torch
import torch.nn as nn
import torch.nn.functional as F

def patch_dinov2_attention_computation(model):
    """Patch DINOv2 attention computation to capture real attention weights"""
    
    def patch_transformer_block(block):
        original_forward = block.forward
        
        def new_forward(x):
            # Store input for residual
            residual = x
            
            # Layer norm
            x_norm = block.norm1(x)
            
            # Get attention module
            attn = block.attn
            
            # Manual attention computation to capture the actual weights
            B, N, C = x_norm.shape
            
            # Get qkv using the actual weights from the attention module
            qkv = attn.qkv(x_norm)
            qkv = qkv.reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            
            # Scale queries
            scale = (C // attn.num_heads) ** -0.5
            q = q * scale
            
            # Compute attention scores
            attn_weights = q @ k.transpose(-2, -1)
            
            # Softmax to get probabilities
            attn_probs = F.softmax(attn_weights, dim=-1)
            
            # Apply dropout if in training mode
            if hasattr(attn, 'attn_drop') and attn.training:
                attn_probs = attn.attn_drop(attn_probs)
            
            # Store in original CLIP format [B * num_heads, seq_len, seq_len]
            B_h, H, L, S = attn_probs.shape
            attn_probs_flat = attn_probs.reshape(B_h * H, L, S)
            block.attn_probs = attn_probs_flat
            
            # Apply attention using the SAME tensor we stored
            out = torch.bmm(block.attn_probs, v.reshape(B_h * H, L, -1))
            out = out.reshape(B_h, H, L, -1).transpose(1, 2).reshape(B_h, L, C)
            
            # Output projection
            if hasattr(attn, 'proj'):
                out = attn.proj(out)
            
            # Apply projection dropout
            if hasattr(attn, 'proj_drop'):
                out = attn.proj_drop(out)
            
            # First residual connection with LayerScale
            if hasattr(block, 'ls1'):
                out = block.ls1(out)
            
            # Apply drop path if exists
            if hasattr(block, 'drop_path1'):
                out = block.drop_path1(out)
            
            x = residual + out
            
            # MLP block
            residual2 = x
            x_norm2 = block.norm2(x)
            mlp_out = block.mlp(x_norm2)
            
            # Second residual connection with LayerScale
            if hasattr(block, 'ls2'):
                mlp_out = block.ls2(mlp_out)
                
            # Apply drop path if exists
            if hasattr(block, 'drop_path2'):
                mlp_out = block.drop_path2(mlp_out)
            
            x = residual2 + mlp_out
            
            return x
        
        block.forward = new_forward
    
    # Patch all transformer blocks
    for block in model.blocks:
        patch_transformer_block(block)

class WrappedDINOv2:
    def __new__(cls, model_name: str, device):
        # Create the original model
        original_model = torch.hub.load('facebookresearch/dinov2', model_name).to(device).eval()
        
        # Change the class of the original model to our wrapper class
        original_model.__class__ = type('WrappedDINOv2', (original_model.__class__,), {
            'enable_attention_capture': cls.enable_attention_capture,
            '_patched': False,
            'model_name': model_name,
            'device': device
        })
        
        # Add wrapper-specific attributes
        original_model._patched = False
        original_model.model_name = model_name
        original_model.device = device
        
        return original_model
    
    def enable_attention_capture(self):
        if not self._patched:
            patch_dinov2_attention_computation(self)
            self._patched = True
        return self

def create_wrapped_dinov2(model_name: str, device):
    return WrappedDINOv2(model_name, device)


import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# Transform setup
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

def compare_dinov2_models(model_name, image_path, device='cuda'):
    """
    Compare original DINOv2 model vs patched version with attention capture
    
    Args:
        model_name: DINOv2 model name (e.g., 'dinov2_vitl14_reg')
        image_path: Path to test image
        device: Device to run on
    """
    
    print(f"=== Comparing Original vs Patched {model_name} ===")
    
    # Create original model
    print("Creating original model...")
    original_model = torch.hub.load('facebookresearch/dinov2', model_name).to(device).eval()
    
    # Create patched model
    print("Creating patched model...")
    patched_model = create_wrapped_dinov2(model_name, device)
    patched_model.enable_attention_capture()
    
    # Prepare test data
    img = Image.open(image_path).convert('RGB')
    img_tensor = dino_transform(img).unsqueeze(0).to(device)
    
    print(f"Input image shape: {img_tensor.shape}")
    
    # Run through original model
    print("\n=== Running original model ===")
    try:
        with torch.no_grad():
            original_features = original_model(img_tensor)
    except RuntimeError as e:
        if "doesn't require gradient" in str(e):
            original_features = original_model(img_tensor)
        else:
            raise e
    
    print(f"Original features shape: {original_features.shape}")
    
    # Run through patched model
    print("\n=== Running patched model ===")
    try:
        with torch.no_grad():
            patched_features = patched_model(img_tensor)
    except RuntimeError as e:
        if "doesn't require gradient" in str(e):
            patched_features = patched_model(img_tensor)
        else:
            raise e
    
    print(f"Patched features shape: {patched_features.shape}")
    
    # Check attention capture
    print("\n=== Checking attention capture ===")
    blocks_with_attn = sum(1 for blk in patched_model.blocks if hasattr(blk, 'attn_probs'))
    total_blocks = len(patched_model.blocks)
    print(f"Blocks with attn_probs: {blocks_with_attn}/{total_blocks}")
    
    if blocks_with_attn > 0:
        first_block = next(blk for blk in patched_model.blocks if hasattr(blk, 'attn_probs'))
        print(f"First block attn_probs shape: {first_block.attn_probs.shape}")
    
    # Detach features for comparison
    original_features = original_features.detach()
    patched_features = patched_features.detach()
    
    # Compare outputs
    print("\n=== Comparing outputs ===")
    
    if original_features.shape == patched_features.shape:
        feature_diff = torch.abs(original_features - patched_features)
        max_diff = feature_diff.max().item()
        mean_diff = feature_diff.mean().item()
        
        print(f"Max difference: {max_diff:.8f}")
        print(f"Mean difference: {mean_diff:.8f}")
        print(f"All close (1e-5): {torch.allclose(original_features, patched_features, atol=1e-5)}")
        print(f"All close (1e-4): {torch.allclose(original_features, patched_features, atol=1e-4)}")
        print(f"All close (1e-3): {torch.allclose(original_features, patched_features, atol=1e-3)}")
        
        if torch.allclose(original_features, patched_features, atol=1e-5):
            print("✓ Models produce identical results!")
            result = "identical"
        elif torch.allclose(original_features, patched_features, atol=1e-3):
            print("≈ Models produce very similar results")
            result = "similar"
        else:
            print("✗ Models produce different results")
            result = "different"
    else:
        print(f"Different output shapes: {original_features.shape} vs {patched_features.shape}")
        result = "incompatible"
    
    return {
        'original_features': original_features,
        'patched_features': patched_features,
        'result': result,
        'attention_blocks': blocks_with_attn
    }

# Convenience function for quick testing
def test_dinov2_patch(image_path, model_name='dinov2_vitl14_reg', device='cuda'):
    """Quick test function"""
    return compare_dinov2_models(model_name, image_path, device)

from types import MethodType

def patch_dinov2_keep_last(model, last_layer: int = 23):
    """
    Keep gradients & attention maps only for transformer *block `last_layer`*.

    Parameters
    ----------
    model : nn.Module
        A DINO-v2 ViT model whose attention blocks already expose
        `.attn_probs` (e.g. after calling `patch_dinov2_attention_computation`).
    last_layer : int, default 23
        Index of the block that should keep full autograd + attention.
    """

    def wrap_block(block, idx):
        orig_forward = block.forward

        def new_forward(self, *args, **kwargs):
            # ---- blocks 0 .. last_layer-1 ---------------------------------
            if idx < last_layer:
                with torch.no_grad():                     # skip tape
                    out = orig_forward(*args, **kwargs)

                # free the attention map (created by the earlier patch)
                if hasattr(self, "attn_probs"):
                    self.attn_probs = None
                return out

            # ---- block == last_layer --------------------------------------
            out = orig_forward(*args, **kwargs)

            # keep grad on the attention map for rollout
            if hasattr(self, "attn_probs"):
                self.attn_probs.retain_grad()
            return out

        # bind the wrapper
        block.forward = MethodType(new_forward, block)

    # Apply to every transformer block inside the DINO backbone
    for i, blk in enumerate(model.blocks):
        wrap_block(blk, i)