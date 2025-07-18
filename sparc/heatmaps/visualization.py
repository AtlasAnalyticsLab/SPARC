"""
Visualization and plotting functions for heatmaps and attributions.

This module provides functions for displaying text attributions, image heatmaps,
and combined visualizations in various formats (HTML, matplotlib plots).
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
from typing import List, Union, Any

from captum.attr import visualization
from IPython.display import HTML, display


def show_heatmap_on_text(
    text: str, 
    text_encoding: torch.Tensor, 
    R_text: torch.Tensor, 
    tokenizer: Any
) -> None:
    """Show heatmap visualization on text using Captum."""
    CLS_idx = text_encoding.argmax(dim=-1)
    R_text = R_text[CLS_idx, 1:CLS_idx]
    text_scores = R_text / R_text.sum()
    text_scores = text_scores.flatten()
    print(text_scores)
    text_tokens = tokenizer.encode(text)
    text_tokens_decoded = [tokenizer.decode([a]) for a in text_tokens]
    vis_data_records = [visualization.VisualizationDataRecord(
        text_scores, 0, 0, 0, 0, 0, text_tokens_decoded, 1
    )]
    visualization.visualize_text(vis_data_records)


def show_image_relevance(
    image_relevance: torch.Tensor, 
    image: torch.Tensor, 
    orig_image: Any
) -> None:
    """Show image relevance heatmap."""
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(orig_image)
    axs[0].axis('off')

    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    axs[1].imshow(vis)
    axs[1].axis('off')


def show_clean_text_attribution(
    text: str, 
    text_encoding: torch.Tensor, 
    R_text: torch.Tensor, 
    tokenizer: Any, 
    show_zero_scores: bool = False,
    min_score_to_show: float = 0.1
) -> str:
    """Show clean text attribution with HTML formatting."""
    CLS_idx = text_encoding.argmax(dim=-1)
    R_text = R_text[CLS_idx, 1:CLS_idx]
    text_scores = R_text / R_text.sum()
    text_scores = text_scores.flatten()
    text_tokens_decoded = [tokenizer.decode([a]) for a in tokenizer.encode(text)]
    scores_norm = (text_scores - text_scores.min()) / (text_scores.max() - text_scores.min())
    
    html = '<div style="line-height: 2; font-size: 16px; font-family: Arial;">'
    latex_code = ""
    
    for token, score in zip(text_tokens_decoded, scores_norm):
        clean_token = token.replace('</w>', ' ').replace('Ġ', ' ').strip()
        if clean_token:
            intensity = float(score)
            score_text = f"{intensity:.2f}"
            
            if intensity > min_score_to_show and score_text != "0.00":
                opacity = intensity 
                html += f'<span style="background: rgba(0,255,0,{opacity}); padding: 2px 4px; margin: 1px; border-radius: 3px;">{clean_token}<sup style="font-size:10px;">{score_text}</sup></span> '
                latex_code += f"\\colorbox{{green!{int(intensity*100)}}}{{{clean_token}}}$^{{{score_text}}}$ "
            else:
                if show_zero_scores and intensity == 0:
                    html += f'<span style="padding: 2px 4px; margin: 1px;">{clean_token}<sup style="font-size:10px;">{score_text}</sup></span> '
                    latex_code += f"{clean_token}$^{{{score_text}}}$ "
                else:
                    html += f'<span style="padding: 2px 4px; margin: 1px;">{clean_token}</span> '
                    latex_code += f"{clean_token} "
    
    html += '</div>'
    display(HTML(html))
    return latex_code


def show_clean_gradcam_text(
    scores: Union[torch.Tensor, np.ndarray], 
    tokens: List[str], 
    show_zero_scores: bool = False
) -> str:
    """Show clean GradCAM text attribution with HTML formatting."""
    if hasattr(scores, 'cpu'):
        scores = scores.cpu().numpy()
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
    
    html = '<div style="line-height: 2; font-size: 16px; font-family: Arial;">'
    latex_code = ""
    
    for token, score in zip(tokens, scores_norm):
        intensity = float(score)
        score_text = f"{intensity:.2f}"
        
        if intensity > 0.05 and score_text != "0.00":
            opacity = intensity * 0.8 + 0.2
            html += f'<span style="background: rgba(0,255,0,{opacity}); padding: 2px 4px; margin: 1px; border-radius: 3px;">{token}<sup style="font-size:10px;">{score_text}</sup></span> '
            latex_code += f"\\colorbox{{green!{int(intensity*80)+10}}}{{{token}}}$^{{{score_text}}}$ "
        else:
            if show_zero_scores:
                html += f'<span style="padding: 2px 4px; margin: 1px;">{token}<sup style="font-size:10px;">{score_text}</sup></span> '
                latex_code += f"{token}$^{{{score_text}}}$ "
            else:
                html += f'<span style="padding: 2px 4px; margin: 1px;">{token}</span> '
                latex_code += f"{token} "
    
    html += '</div>'
    display(HTML(html))
    return latex_code


def plot_all_attributions(
    image: Any, 
    cam_img_dino_latent: np.ndarray, 
    cam_image_clip_latent: np.ndarray, 
    cam_image_clip_sim: np.ndarray,
    sparc_dino_relevance: torch.Tensor, 
    sparc_clip_img_relevance: torch.Tensor, 
    clip_img_sim_relevance: torch.Tensor, 
    show_titles: bool = True,
    title_fontsize: int = 10,
    title_fontfamily: str = 'sans-serif',
    title_fontweight: str = 'normal'
) -> plt.Figure:
    """Plot all attribution visualizations in a single figure."""
    def create_heatmap_overlay(img, relevance_tensor):
        dim = int(relevance_tensor.numel() ** 0.5)
        image_relevance = relevance_tensor.reshape(1, 1, dim, dim)
        image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
        image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
        image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
                
        heatmap = cv2.applyColorMap(np.uint8(255 * image_relevance), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)/255
        cam = cam / np.max(cam)
        vis = np.uint8(255 * cam)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        return vis
    
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(1, 7, wspace=0.01, hspace=0.3)
    axes = []
    for row in range(1):
        for col in range(7):
            axes.append(fig.add_subplot(gs[row, col]))
    
    original_resized = image.resize((224, 224))
    img = np.array(original_resized)
    dino_overlay = create_heatmap_overlay(img, sparc_dino_relevance)
    clip_overlay = create_heatmap_overlay(img, sparc_clip_img_relevance) 
    r_image_overlay = create_heatmap_overlay(img, clip_img_sim_relevance)
    
    images = [
        original_resized, dino_overlay, clip_overlay, cam_img_dino_latent, 
        cam_image_clip_latent, r_image_overlay, cam_image_clip_sim
    ]
    
    titles = [
        "Original", "Relevancy Map\nDINO (SPARC)", "Relevancy Map\nCLIP (SPARC)", 
        "GradCAM\nDINO (SPARC)", "GradCAM\nCLIP (SPARC)", "Relevancy Map\nCLIP Sim", "GradCAM\nCLIP Sim"
    ]
    
    for i, (ax, img, title) in enumerate(zip(axes, images, titles)):
        ax.imshow(img)
        if show_titles:
            ax.set_title(title, fontsize=title_fontsize, fontfamily=title_fontfamily, fontweight=title_fontweight)
        ax.axis('off')
    return fig


def plot_relevancy_attributions(
    image: Any, 
    sparc_dino_relevance: torch.Tensor, 
    sparc_clip_img_relevance: torch.Tensor, 
    clip_img_sim_relevance: torch.Tensor, 
    show_titles: bool = True,
    title_fontsize: int = 10,
    title_fontfamily: str = 'sans-serif',
    title_fontweight: str = 'normal'
) -> plt.Figure:
    """Plot only relevancy attribution visualizations."""
    def create_heatmap_overlay(img, relevance_tensor):
        dim = int(relevance_tensor.numel() ** 0.5)
        image_relevance = relevance_tensor.reshape(1, 1, dim, dim)
        image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
        image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
        image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
                
        heatmap = cv2.applyColorMap(np.uint8(255 * image_relevance), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)/255
        cam = cam / np.max(cam)
        vis = np.uint8(255 * cam)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        return vis
    
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(1, 4, wspace=0.01, hspace=0.3)
    axes = []
    for row in range(1):
        for col in range(4):
            axes.append(fig.add_subplot(gs[row, col]))
    
    original_resized = image.resize((224, 224))
    img = np.array(original_resized)
    dino_overlay = create_heatmap_overlay(img, sparc_dino_relevance)
    clip_overlay = create_heatmap_overlay(img, sparc_clip_img_relevance) 
    r_image_overlay = create_heatmap_overlay(img, clip_img_sim_relevance)
    
    images = [original_resized, dino_overlay, clip_overlay, r_image_overlay]
    titles = ["Original", "Relevancy Map\nDINO (SPARC)", "Relevancy Map\nCLIP (SPARC)", "Relevancy Map\nCLIP Sim"]
    
    for i, (ax, img, title) in enumerate(zip(axes, images, titles)):
        ax.imshow(img)
        if show_titles:
            ax.set_title(title, fontsize=title_fontsize, fontfamily=title_fontfamily, fontweight=title_fontweight)
        ax.axis('off')
    return fig 