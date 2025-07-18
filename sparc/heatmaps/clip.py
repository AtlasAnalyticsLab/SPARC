import torch
from PIL import Image

import torch.nn.functional as F
import open_clip


def patch_attention_computation(model):
    """Patch the actual attention computation to capture real attention weights"""
    
    def patch_transformer_block(block):
        original_forward = block.forward
        
        def new_forward(x, attn_mask=None):
            # Store input for residual
            residual = x
            
            # Layer norm
            x = block.ln_1(x)
            
            # Get attention module
            attn = block.attn
            
            # Manual attention computation to capture the actual weights
            B, N, C = x.shape
            
            # Get qkv using the actual weights from the attention module
            if hasattr(attn, 'in_proj_weight'):
                # Standard MultiheadAttention with combined qkv projection
                qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
                qkv = qkv.reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
            else:
                # Separate q, k, v projections (fallback)
                q = F.linear(x, attn.q_proj_weight, getattr(attn, 'q_proj_bias', None))
                k = F.linear(x, attn.k_proj_weight, getattr(attn, 'k_proj_bias', None))  
                v = F.linear(x, attn.v_proj_weight, getattr(attn, 'v_proj_bias', None))
                
                # Reshape for multi-head attention
                q = q.reshape(B, N, attn.num_heads, C // attn.num_heads).transpose(1, 2)
                k = k.reshape(B, N, attn.num_heads, C // attn.num_heads).transpose(1, 2)
                v = v.reshape(B, N, attn.num_heads, C // attn.num_heads).transpose(1, 2)
            
            # Scale queries
            scale = (C // attn.num_heads) ** -0.5
            q = q * scale
            
            # Compute attention scores
            attn_weights = q @ k.transpose(-2, -1)
            
            # Apply attention mask if provided
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_weights.masked_fill_(attn_mask, float('-inf'))
                else:
                    attn_weights += attn_mask
            
            # Softmax to get probabilities
            attn_probs = F.softmax(attn_weights, dim=-1)
            
            # Apply dropout if in training mode
            if hasattr(attn, 'dropout') and attn.training:
                attn_probs = F.dropout(attn_probs, p=attn.dropout, training=True)
            
            # Store in original CLIP format [B * num_heads, seq_len, seq_len]
            B_h, H, L, S = attn_probs.shape
            attn_probs_flat = attn_probs.reshape(B_h * H, L, S)
            block.attn_probs = attn_probs_flat
            
            # Apply attention using the SAME tensor we stored
            out = torch.bmm(block.attn_probs, v.reshape(B_h * H, L, -1))
            out = out.reshape(B_h, H, L, -1).transpose(1, 2).reshape(B_h, L, C)
            
            # Output projection
            if hasattr(attn, 'out_proj'):
                out = attn.out_proj(out)
            
            # First residual connection
            x = residual + out
            
            # MLP block
            x = x + block.mlp(block.ln_2(x)) 
            
            return x
        
        block.forward = new_forward
    
    # Patch vision transformer blocks
    if hasattr(model, 'visual') and hasattr(model.visual, 'transformer'):
        for block in model.visual.transformer.resblocks:
            patch_transformer_block(block)
    
    # Patch text transformer blocks
    if hasattr(model, 'transformer'):
        for block in model.transformer.resblocks:
            patch_transformer_block(block)

class WrappedOpenCLIP:
    def __new__(cls, model_name: str, device):
        # Create the original model
        original_model, _, preprocess = open_clip.create_model_and_transforms(model_name)
        original_model = original_model.to(device)
        
        # Change the class of the original model to our wrapper class
        original_model.__class__ = type('WrappedOpenCLIP', (original_model.__class__,), {
            'enable_attention_capture': cls.enable_attention_capture,
            '_patched': False,
            'preprocess': preprocess
        })
        
        # Add wrapper-specific attributes
        original_model._patched = False
        original_model.preprocess = preprocess
        
        return original_model
    
    def enable_attention_capture(self):
        if not self._patched:
            patch_attention_computation(self)
            self._patched = True
        return self

def create_wrapped_clip(model_name: str, device):
    return WrappedOpenCLIP(model_name, device)


def compare_clip_models(
    model1, model2,
    preprocess1, preprocess2,
    tokenizer1, tokenizer2,
    image_path, text_list,
    device='cuda',
    model1_name="Model 1",
    model2_name="Model 2"
):
    """
    Compare two CLIP models on the same data.
    
    Args:
        model1, model2: The models to compare
        preprocess1, preprocess2: Image preprocessing functions for each model
        tokenizer1, tokenizer2: Text tokenizers for each model
        image_path: Path to test image
        text_list: List of text strings to test
        device: Device to run on
        model1_name, model2_name: Names for display
    """
    
    print(f"=== Comparing {model1_name} vs {model2_name} ===")
    
    # Prepare image data
    img = Image.open(image_path)
    img1 = preprocess1(img).unsqueeze(0).to(device)
    img2 = preprocess2(img).unsqueeze(0).to(device)
    
    # Prepare text data
    text1 = tokenizer1(text_list).to(device)
    text2 = tokenizer2(text_list).to(device)
    
    batch_size = len(text_list)
    images1 = img1.repeat(batch_size, 1, 1, 1)
    images2 = img2.repeat(batch_size, 1, 1, 1)
    
    print(f"Testing with {len(text_list)} texts: {text_list}")
    print(f"Batch size: {batch_size}")
    print(f"{model1_name} - Images shape: {images1.shape}, Text shape: {text1.shape}")
    print(f"{model2_name} - Images shape: {images2.shape}, Text shape: {text2.shape}")
    
    # Set models to eval mode
    model1.eval()
    model2.eval()
    
    # Run through first model
    print(f"\n=== Running {model1_name} ===")
    try:
        with torch.no_grad():
            if hasattr(model1, 'encode_image'):
                features1_img = model1.encode_image(images1)
                features1_text = model1.encode_text(text1)
            else:
                # Handle different model interfaces
                features1_img = model1.visual(images1)
                features1_text = model1.text(text1)
    except RuntimeError as e:
        if "doesn't require gradient" in str(e):
            # Model needs gradients enabled (like custom CLIP with hooks)
            if hasattr(model1, 'encode_image'):
                features1_img = model1.encode_image(images1)
                features1_text = model1.encode_text(text1)
            else:
                features1_img = model1.visual(images1)
                features1_text = model1.text(text1)
        else:
            raise e
    
    print(f"{model1_name} - Image features shape: {features1_img.shape}")
    print(f"{model1_name} - Text features shape: {features1_text.shape}")
    
    # Run through second model
    print(f"\n=== Running {model2_name} ===")
    try:
        with torch.no_grad():
            if hasattr(model2, 'encode_image'):
                features2_img = model2.encode_image(images2)
                features2_text = model2.encode_text(text2)
            else:
                # Handle different model interfaces
                features2_img = model2.visual(images2)
                features2_text = model2.text(text2)
    except RuntimeError as e:
        if "doesn't require gradient" in str(e):
            # Model needs gradients enabled (like custom CLIP with hooks)
            if hasattr(model2, 'encode_image'):
                features2_img = model2.encode_image(images2)
                features2_text = model2.encode_text(text2)
            else:
                features2_img = model2.visual(images2)
                features2_text = model2.text(text2)
        else:
            raise e
    
    print(f"{model2_name} - Image features shape: {features2_img.shape}")
    print(f"{model2_name} - Text features shape: {features2_text.shape}")
    
    # Detach features for comparison (in case gradients are enabled)
    features1_img = features1_img.detach()
    features1_text = features1_text.detach()
    features2_img = features2_img.detach()
    features2_text = features2_text.detach()
    
    # Check if models have attention capture capability
    print(f"\n=== Checking attention capture ===")
    
    def check_attention_capture(model, name):
        if hasattr(model, 'visual') and hasattr(model.visual, 'transformer'):
            vision_blocks = list(model.visual.transformer.resblocks)
        elif hasattr(model, 'model') and hasattr(model.model.visual, 'transformer'):
            vision_blocks = list(model.model.visual.transformer.resblocks)
        else:
            vision_blocks = []
        
        if hasattr(model, 'transformer'):
            text_blocks = list(model.transformer.resblocks)
        elif hasattr(model, 'model') and hasattr(model.model, 'transformer'):
            text_blocks = list(model.model.transformer.resblocks)
        else:
            text_blocks = []
        
        vision_with_attn = sum(1 for blk in vision_blocks if hasattr(blk, 'attn_probs'))
        text_with_attn = sum(1 for blk in text_blocks if hasattr(blk, 'attn_probs'))
        
        print(f"{name} - Vision blocks with attn_probs: {vision_with_attn}/{len(vision_blocks)}")
        print(f"{name} - Text blocks with attn_probs: {text_with_attn}/{len(text_blocks)}")
        
        if vision_blocks and hasattr(vision_blocks[0], 'attn_probs'):
            print(f"{name} - First vision attn_probs shape: {vision_blocks[0].attn_probs.shape}")
        if text_blocks and hasattr(text_blocks[0], 'attn_probs'):
            print(f"{name} - First text attn_probs shape: {text_blocks[0].attn_probs.shape}")
    
    check_attention_capture(model1, model1_name)
    check_attention_capture(model2, model2_name)
    
    # Compare feature outputs
    print(f"\n=== Comparing outputs ===")
    
    # Image features comparison
    if features1_img.shape == features2_img.shape:
        img_diff = torch.abs(features1_img - features2_img)
        img_max_diff = img_diff.max().item()
        img_mean_diff = img_diff.mean().item()
        
        print(f"Image features - Max difference: {img_max_diff:.8f}")
        print(f"Image features - Mean difference: {img_mean_diff:.8f}")
        print(f"Image features - All close (1e-5): {torch.allclose(features1_img, features2_img, atol=1e-5)}")
        print(f"Image features - All close (1e-4): {torch.allclose(features1_img, features2_img, atol=1e-4)}")
        print(f"Image features - All close (1e-3): {torch.allclose(features1_img, features2_img, atol=1e-3)}")
    else:
        print(f"Image features - Different shapes: {features1_img.shape} vs {features2_img.shape}")
    
    # Text features comparison
    if features1_text.shape == features2_text.shape:
        text_diff = torch.abs(features1_text - features2_text)
        text_max_diff = text_diff.max().item()
        text_mean_diff = text_diff.mean().item()
        
        print(f"Text features - Max difference: {text_max_diff:.8f}")
        print(f"Text features - Mean difference: {text_mean_diff:.8f}")
        print(f"Text features - All close (1e-5): {torch.allclose(features1_text, features2_text, atol=1e-5)}")
        print(f"Text features - All close (1e-4): {torch.allclose(features1_text, features2_text, atol=1e-4)}")
        print(f"Text features - All close (1e-3): {torch.allclose(features1_text, features2_text, atol=1e-3)}")
    else:
        print(f"Text features - Different shapes: {features1_text.shape} vs {features2_text.shape}")
    
    # Compute and compare similarities
    print(f"\n=== Computing similarities ===")
    
    # Normalize features
    features1_img_norm = features1_img / features1_img.norm(dim=-1, keepdim=True)
    features1_text_norm = features1_text / features1_text.norm(dim=-1, keepdim=True)
    features2_img_norm = features2_img / features2_img.norm(dim=-1, keepdim=True)
    features2_text_norm = features2_text / features2_text.norm(dim=-1, keepdim=True)
    
    # Compute similarity matrices
    similarities1 = (features1_img_norm @ features1_text_norm.T) * 100
    similarities2 = (features2_img_norm @ features2_text_norm.T) * 100
    
    print(f"{model1_name} similarities:")
    print(similarities1.cpu().numpy())
    print(f"\n{model2_name} similarities:")
    print(similarities2.cpu().numpy())
    
    if similarities1.shape == similarities2.shape:
        sim_diff = torch.abs(similarities1 - similarities2)
        print(f"\nSimilarity difference - Max: {sim_diff.max().item():.6f}")
        print(f"Similarity difference - Mean: {sim_diff.mean().item():.6f}")
        
        if torch.allclose(similarities1, similarities2, atol=1e-3):
            print("✓ Models produce identical results!")
            result = "identical"
        elif torch.allclose(similarities1, similarities2, atol=1e-2):
            print("≈ Models produce very similar results (within 1e-2)")
            result = "very_similar"
        elif torch.allclose(similarities1, similarities2, atol=1e-1):
            print("~ Models produce similar results (within 1e-1)")
            result = "similar"
        else:
            print("✗ Models produce different results")
            result = "different"
    else:
        print("Cannot compare similarities - different output shapes")
        result = "incomparable"
    
    return {
        'image_features_1': features1_img,
        'text_features_1': features1_text,
        'image_features_2': features2_img,
        'text_features_2': features2_text,
        'similarities_1': similarities1,
        'similarities_2': similarities2,
        'result': result
    }

# Convenience function for your specific use case
def compare_original_vs_wrapped(clip_model_name, image_path, text_list, device='cuda'):
    """
    Compare original open_clip model vs wrapped version with attention capture
    """
    
    # Create original model
    print("Creating original model...")
    original_model, _, preprocess_orig = open_clip.create_model_and_transforms(clip_model_name)
    original_model = original_model.to(device)
    tokenizer_orig = open_clip.get_tokenizer(clip_model_name)
    
    # Create wrapped model
    print("Creating wrapped model...")
    wrapped_model = create_wrapped_clip(clip_model_name, device)
    wrapped_model.enable_attention_capture()
    preprocess_wrapped = wrapped_model.preprocess
    tokenizer_wrapped = open_clip.get_tokenizer(clip_model_name)  # Same tokenizer
    
    return compare_clip_models(
        original_model, wrapped_model,
        preprocess_orig, preprocess_wrapped,
        tokenizer_orig, tokenizer_wrapped,
        image_path, text_list, device,
        "Original OpenCLIP", "Wrapped OpenCLIP"
    )



from types import MethodType

def patch_clip_keep_last(model, last_layer: int = 23):
    """
    Keep gradients & attention maps only for `last_layer` in each CLIP stack.

    Call this *after* `patch_attention_computation(model)` (so every block
    already writes `block.attn_probs`).  It works for OpenAI CLIP, OpenCLIP,
    and similar ViT-based variants.

    Parameters
    ----------
    model : nn.Module
        Your wrapped CLIP model.
    last_layer : int, default 23
        Transformer block index that should keep full autograd
        (0-based: 0 … 11 for ViT-B/32, 0 … 23 for ViT-B/16/B/32-XL, etc.).
    """

    def wrap_block(block, idx):
        orig_forward = block.forward          # already patched by attention-capture

        def new_forward(self, *args, **kwargs):
            # -------- blocks 0 .. last_layer-1 ----------------------------
            if idx < last_layer:
                with torch.no_grad():         # skip autograd tape
                    out = orig_forward(*args, **kwargs)

                # drop the stored attention map to free GPU RAM
                if hasattr(self, "attn_probs"):
                    self.attn_probs = None
                return out

            # -------- block == last_layer ---------------------------------
            out = orig_forward(*args, **kwargs)

            # keep gradient on attn_probs for rollout
            if hasattr(self, "attn_probs"):
                self.attn_probs.retain_grad()
            return out

        block.forward = MethodType(new_forward, block)

    # ---------- image tower ------------------------------------------------
    if hasattr(model, "visual") and hasattr(model.visual, "transformer"):
        for i, blk in enumerate(model.visual.transformer.resblocks):
            wrap_block(blk, i)

    # ---------- text tower -------------------------------------------------
    if hasattr(model, "transformer"):                  # OpenCLIP / OpenAI CLIP
        for i, blk in enumerate(model.transformer.resblocks):
            wrap_block(blk, i)