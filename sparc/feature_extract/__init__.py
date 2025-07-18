"""
Feature extraction module for SPARC

Contains utilities for extracting features from COCO and Open Images datasets.
"""

from .extract_coco import (
    extract_dino_features_coco,
    extract_clip_features_coco,
    CocoImagesDataset,
)

from .extract_open_images import (
    extract_dino_features_open_images,
    extract_clip_features_open_images,
    OpenImagesDataset,
)

__all__ = [
    "extract_dino_features_coco",
    "extract_clip_features_coco",
    "CocoImagesDataset", 
    "extract_dino_features_open_images",
    "extract_clip_features_open_images",
    "OpenImagesDataset",
] 