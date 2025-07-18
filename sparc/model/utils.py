import torch
import torch.nn as nn

# --- Model Utility Functions ---
def unit_norm_decoder_grad_adjustment_(decoder: nn.Linear):
    """Project out gradient component parallel to the unit vectors"""
    if decoder.weight.grad is None:
        return
    # Calculate dot product of weight and its gradient
    parallel_component = torch.sum(
        decoder.weight.data * decoder.weight.grad, 
        dim=0, keepdim=True
    )
    # Subtract the parallel component
    decoder.weight.grad -= parallel_component * decoder.weight.data
    
# --- Unit Norm Utility Functions ---
def unit_norm_decoder_(decoder: nn.Linear):
    """Unit normalize the decoder weights column-wise."""
    decoder.weight.data /= (decoder.weight.data.norm(dim=0, keepdim=True) + 1e-8)
