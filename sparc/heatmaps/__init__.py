# """
# Heatmap visualization module for multimodal models (CLIP, DINO) and sparse autoencoders.

# This module provides tools for generating various types of attention visualizations:
# - GradCAM visualizations for vision models
# - Text attribution using gradient-based methods
# - Attention rollout for transformer models
# - Integration with sparse autoencoders (SAE/MSAE)

# The module is organized into several submodules:
# - transforms: Reshape functions for different model types
# - tokenization: Token conversion and word merging utilities
# - models: Model wrapper classes and dataset utilities
# - gradcam: GradCAM visualization functions
# - text_attribution: Text attribution methods
# - attention_rollout: Attention rollout implementations
# - visualization: Plotting and display functions
# """

# # Import all main functions and classes for easy access
# from .transforms import (
#     reshape_transform_clip,
#     reshape_transform_dino,
#     reshape_transform_text
# )

# from .tokenization import (
#     ids_to_tokens,
#     tokens_to_words
# )

# from .models import (
#     CocoImagesDataset,
#     TextGuidedSAE,
#     TextGuidedClip
# )

# from .gradcam import (
#     visualize_cam,
#     compute_gradcam
# )

# from .text_attribution import (
#     gradtext_lastlayer,
#     gradtext
# )

# from .attention_rollout import (
#     get_attention_blocks,
#     compute_attention_relevancy,
#     get_all_latents,
#     interpret_clip,
#     interpret_sparc
# )

# from .visualization import (
#     show_heatmap_on_text,
#     show_image_relevance,
#     show_clean_text_attribution,
#     show_clean_gradcam_text,
#     plot_all_attributions,
#     plot_relevancy_attributions
# )

# # Define what gets imported with "from heatmaps import *"
# __all__ = [
#     # Transform functions
#     'reshape_transform_clip',
#     'reshape_transform_dino', 
#     'reshape_transform_text',
    
#     # Tokenization utilities
#     'ids_to_tokens',
#     'tokens_to_words',
    
#     # Model classes
#     'CocoImagesDataset',
#     'TextGuidedSAE',
#     'TextGuidedClip',
    
#     # GradCAM functions
#     'visualize_cam',
#     'compute_gradcam',
    
#     # Text attribution functions
#     'gradtext_lastlayer',
#     'gradtext',
    
#     # Attention rollout functions
#     'get_attention_blocks',
#     'compute_attention_relevancy',
#     'get_all_latents',
#     'interpret_clip',
#     'interpret_sparc',
    
#     # Visualization functions
#     'show_heatmap_on_text',
#     'show_image_relevance',
#     'show_clean_text_attribution',
#     'show_clean_gradcam_text',
#     'plot_all_attributions',
#     'plot_relevancy_attributions'
# ]

# # Import everything (backward compatible)
# from heatmaps import compute_gradcam, interpret_sparc, plot_all_attributions

# # Import specific modules
# from heatmaps.gradcam import visualize_cam
# from heatmaps.text_attribution import gradtext_lastlayer
# from heatmaps.visualization import show_clean_text_attribution