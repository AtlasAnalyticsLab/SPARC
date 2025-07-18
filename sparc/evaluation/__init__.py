"""
Evaluation module for SPARC

Contains various evaluation metrics and methods for assessing model performance
across different tasks including classification, retrieval, and concept alignment.
"""

from .classification import (
    eval_classification,
    run_downstream_classification_eval,
    save_classification_results,
    mean_average_precision,
    MultiLabelClassifier,
    LatentDataset,
    BaseLabelProvider,
    CocoLabelProvider,
    OpenImagesLabelProvider,
)

from .retrieval import (
    eval_retrieval,
    run_downstream_retrieval_eval,
)

# from .concept_alignment import (
#     save_concept_alignment_results,
#     assign_concepts_to_latents,
#     inspect_latent,
#     compute_concept_alignment,
# )

# from .latent_alignment import (
#     compute_linear_cka,
#     print_cka,
# )

# from .cross_stream_metrics import (
#     compute_cross_reconstruction_fidelity,
# )

# from .concept_metric import (
#     compute_concept_overlap,
# )

# from .auprc_profile_correlation import (
#     compute_auprc_profile_correlation,
#     print_auprc_correlation,
# )

__all__ = [
    "eval_classification",
    "run_downstream_classification_eval", 
    "save_classification_results",
    "mean_average_precision",
    "MultiLabelClassifier",
    "LatentDataset",
    "BaseLabelProvider",
    "CocoLabelProvider", 
    "OpenImagesLabelProvider",
    "eval_retrieval",
    "run_downstream_retrieval_eval",
] 