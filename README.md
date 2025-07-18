# SPARC: Concept-Aligned Sparse Autoencoders for Cross-Model and Cross-Modal Interpretability

**SPARC** (**Sp**arse Autoencoders for **A**ligned **R**epresentation of **C**oncepts) learns a unified latent space shared across diverse models (DINO, CLIP Vision, CLIP Text) where individual dimensions correspond to the same concepts across different architectures and modalities.

## Installation

```bash
git clone https://github.com/AtlasAnalyticsLab/SPARC.git
cd SPARC
pip install -e .
```

## Pre-computed Features & Results

Download pre-extracted features and trained models:

**[Download Features & Results](https://drive.google.com/drive/folders/1DRvYml0rF5ZRpwCVpdDvVoI8TiU4aUzl)**

Contains:
- `features/` - Pre-extracted DINO and CLIP features for Open Images and COCO
- `final_results/` - Pre-trained SPARC model weights and analysis results

## Scripts & Notebooks

### Main Scripts

- **`main.py`** - Train SPARC models or run evaluation (classification, retrieval)
- **`sparc/feature_extract/extract_coco.py`** - Extract DINO/CLIP features from COCO dataset  
- **`sparc/feature_extract/extract_open_images.py`** - Extract DINO/CLIP features from Open Images

### Qualitative Analysis Notebooks

- **`notebooks/qual/t2_aligned_concept_visualization.ipynb`** - Visualize concept alignment across models
- **`notebooks/qual/t3_heatmaps_from_caption.ipynb`** - Generate cross-modal attribution heatmaps
- **`notebooks/qual/scripts_all_results/t2_concept_batch_visualization.py`** - Batch concept visualization script
- **`notebooks/qual/scripts_all_results/t3_caption_visualization.py`** - Caption-based visualization script

### Quantitative Analysis Notebooks

- **`notebooks/quant/t3_segmentation-coco-latents-global.ipynb`** - Semantic segmentation evaluation (Global TopK)
- **`notebooks/quant/t3_segmentation-coco-latents-local.ipynb`** - Semantic segmentation evaluation (Local TopK)  
- **`notebooks/quant/t3_segmentation-coco-caption.ipynb`** - Caption-based segmentation evaluation

## Visualizations

Pre-computed visualization examples are available:

- **[Latent Visualizations](https://drive.google.com/drive/folders/1fTSPH9BiBoNzMroSia8TG7hxC1wQpfMX)** - SPARC latent aligned examples
- **[Single Concept Visualizations](https://drive.google.com/drive/folders/1_Dl5PjRSRSj7SHP5otDiWES1WjLI8stT)** - Concept-specific attribution examples  
- **[Caption Visualizations](https://drive.google.com/drive/folders/1t4wtK-63DrbLBU5QMHS6TAqi7SkFArvO)** - Cross-modal attribution examples

## Quick Example

```bash
# Train SPARC model
python main.py --config configs/config_coco.json --topk_type global --n_latents 8192 --k 64 --cross_loss_coef 1.0

# Run evaluation only (with pre-trained model)
python main.py --config configs/config_coco.json --only_eval
```

## Citation

```bibtex
@misc{nasirisarvi2025sparcconceptalignedsparseautoencoders,
      title={SPARC: Concept-Aligned Sparse Autoencoders for Cross-Model and Cross-Modal Interpretability}, 
      author={Ali Nasiri-Sarvi and Hassan Rivaz and Mahdi S. Hosseini},
      year={2025},
      eprint={2507.06265},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.06265}, 
}
```

## Contact

- GitHub: [AtlasAnalyticsLab/SPARC](https://github.com/AtlasAnalyticsLab/SPARC)
- Email: ali.nasirisarvi@mail.concordia.ca