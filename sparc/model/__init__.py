from .model_global import MultiStreamSparseAutoencoder as _MultiStreamSparseAutoencoderGlobal
from .model_local import MultiStreamSparseAutoencoder as _MultiStreamSparseAutoencoderLocal

from typing import Type, Literal
import torch.nn as nn

# Type hint for the model classes
SAEModelType = Type[nn.Module] 

def get_sae_model_class(topk_type: Literal['global', 'local']) -> SAEModelType:
    """
    Returns the appropriate MultiStreamSparseAutoencoder class based on topk_type.

    Args:
        topk_type: Specifies whether to use 'global' or 'local' top-k selection.

    Returns:
        The corresponding model class (_MultiStreamSparseAutoencoderGlobal or 
        _MultiStreamSparseAutoencoderLocal).
        
    Raises:
        ValueError: If topk_type is not 'global' or 'local'.
    """
    if topk_type == 'global':
        print("Selecting Global TopK MultiStreamSparseAutoencoder class")
        return _MultiStreamSparseAutoencoderGlobal
    elif topk_type == 'local':
        print("Selecting Local TopK MultiStreamSparseAutoencoder class")
        return _MultiStreamSparseAutoencoderLocal
    else:
        raise ValueError(f"Invalid topk_type: {topk_type}. Must be 'global' or 'local'.")