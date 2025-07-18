import os
import json
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

import random
import textwrap


from sparc.datasets import HDF5FeatureDataset
from sparc.model import get_sae_model_class
from sparc.post_analysis import HDF5AnalysisResultsDataset
from sparc.feature_extract.extract_open_images import OpenImagesDataset


def visualize_multi_stream_top_activating_images(
    dim_idx: int, 
    analysis_results: dict, 
    dataset,
    n_latents: int, 
    num_images: int = 5 # Number of images per modality row
):
    """
    Visualize top activating images for a specific dimension across all three streams 
    (DINO, CLIP Image, CLIP Text) from the analysis results. Displays text captions separately.

    Args:
        dim_idx: The latent dimension index to visualize.
        analysis_results: Dictionary returned by return_top_activating_images.
        dataset: OpenImagesDataset to retrieve images and text captions.
        n_latents: Total number of latent dimensions (for validation).
        num_images: Number of top images to display per modality. Defaults to 5.
    """
    if not (0 <= dim_idx < n_latents):
        print(f"Error: dim_idx ({dim_idx}) must be between 0 and {n_latents-1}")
        return
        
    if num_images <= 0:
        print("Error: num_images must be positive.")
        return

    modalities = ['dino', 'clip_img', 'clip_txt']
    
    # --- Plotting Setup ---
    fig, axes = plt.subplots(len(modalities), num_images, 
                             figsize=(num_images * 3, len(modalities) * 3 + 1)) # Adjust size +1 for captions
    fig.suptitle(f"Top {num_images} Activating Images for Dimension {dim_idx}", fontsize=16, y=0.98)

    if axes.ndim == 1: # Handle case where num_images=1 or len(modalities)=1
        axes = axes.reshape(-1, num_images if len(modalities) > 1 else 1)
        if axes.shape[0] == 1 and len(modalities)>1: # if num_images=1, make it col vector
             axes = axes.T

    text_captions_info = [] # To store caption info for text modality

    # --- Display Images ---
    for row, mod in enumerate(modalities):
        if mod not in analysis_results.streams:
            print(f"Warning: Modality '{mod}' not found in analysis_results. Skipping row.")
            for col in range(num_images):
                if axes.shape[0] > row and axes.shape[1] > col:
                     axes[row, col].text(0.5, 0.5, f"'{mod}' data\nnot found", ha='center', va='center', fontsize=10)
                     axes[row, col].axis('off')
            continue
            
        # Get top activating images/indices for this dimension and modality
        top_acts = analysis_results.get_top_activations_for_latent(mod, dim_idx)[:num_images]

        for col, (activation, img_idx) in enumerate(top_acts):
            if col >= num_images: break 
            img_idx = int(img_idx)
            ax = axes[row, col]
            
            try:
                # Fetch image data (assuming it's the first element)
                img = dataset[img_idx]['image'] 
                ax.imshow(img)
                ax.set_title(f"{mod.upper()} Act: {activation:.2f}", fontsize=9)
                ax.axis('off')

                # If it's the text modality row, store info needed for caption display later
                if mod == 'clip_txt':
                     # Fetch the first caption if available
                     caption = "Caption not found"
                     if len(dataset[img_idx]) > 1 and len(dataset[img_idx]['captions']) > 0:
                          caption = dataset[img_idx]['captions']
                     text_captions_info.append({
                          "col": col, 
                          "img_idx": img_idx, 
                          "activation": activation, 
                          "caption": caption
                     })

            except KeyError:
                 ax.text(0.5, 0.5, f"Image Idx {img_idx}\nnot in open_imgs", ha='center', va='center', fontsize=10)
                 ax.axis('off')
                 print(f"Error: Image index {img_idx} not found in open_imgs for dim {dim_idx}, mod {mod}.")
            except Exception as e:
                 ax.text(0.5, 0.5, f"Error loading\nimage {img_idx}", ha='center', va='center', fontsize=10)
                 ax.axis('off')
                 print(f"Error loading image {img_idx} for dim {dim_idx}, mod {mod}: {e}")
                 
        # Fill remaining columns if fewer than num_images were found
        for col in range(len(top_acts), num_images):
             ax = axes[row, col]
             ax.text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=10)
             ax.axis('off')

    # --- Display Text Captions Separately Below ---
    caption_text_lines = [f"Captions for TEXT modality (Dim {dim_idx}):"]
    if text_captions_info:
         # Sort by column appearance just in case order was disrupted
         text_captions_info.sort(key=lambda item: item['col']) 
         for item in text_captions_info:
              # Wrap caption text
              wrapped_caption = textwrap.fill(item['caption'], width=200) # Adjust width as needed
              caption_text_lines.append(f" Img {item['col']+1} (Idx:{item['img_idx']}, Act:{item['activation']:.2f}): {wrapped_caption}")
    else:
         caption_text_lines.append(" (No text modality images/captions found or processed)")

    full_caption_text = "\n".join(caption_text_lines)
    
    # Adjust y position (0.01 is near bottom), fontsize, and alignment as needed
    plt.figtext(0.01, 0.01, full_caption_text, ha="left", va="bottom", fontsize=10, wrap=False)

    # Adjust layout to prevent overlap and make space for captions
    plt.tight_layout(rect=[0, 0.2, 1, 0.96]) # rect=[left, bottom, right, top] - bottom=0.1 leaves space

    plt.show()


def create_complete_figure(dim_idx, scenarios_list, dataset, num_images=10, img_size=140, random_sample=False, center_crop=False):
    modalities = ['dino', 'clip_img', 'clip_txt']
    modality_labels = ['Stream A (DINO)', 'Stream B (CLIP-img)', 'Stream C (CLIP-txt)']
    scenario_labels = [
        'Local TopK, λ=0',
        'Local TopK, λ=1',
        'Global TopK, λ=0',
        'Global TopK, λ=1'
    ]
    
    def center_crop_image(img, target_size):
        """
        Center crop the image to a square and resize to target_size.
        
        Args:
            img: PIL Image object
            target_size: Target size for the final image (int)
        
        Returns:
            PIL Image object that has been center cropped and resized
        """
        if not hasattr(img, 'size'):
            return img
        
        width, height = img.size
        
        # Handle edge case where image has zero dimensions
        if width <= 0 or height <= 0:
            return img
        
        # Calculate the size of the square crop (minimum dimension)
        crop_size = min(width, height)
        
        # Handle case where crop_size is zero
        if crop_size <= 0:
            return img
        
        # Calculate crop coordinates to center the crop
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        
        # Ensure coordinates are within bounds
        left = max(0, left)
        top = max(0, top)
        right = min(width, right)
        bottom = min(height, bottom)
        
        # Crop to square
        img_cropped = img.crop((left, top, right, bottom))
        
        # Resize to target size
        if target_size > 0:
            img_resized = img_cropped.resize((target_size, target_size))
            return img_resized
        else:
            return img_cropped
    
    # Create compact 2x2 figure with minimal spacing between blocks
    base_width = 27
    scaled_width = base_width * (num_images / 10)
    
    fig = plt.figure(figsize=(scaled_width, 9), dpi=300)
    fig.text(0.07, 0.94, f"Latent Dimension {dim_idx}", fontsize=12, fontweight='bold', 
         fontname='monospace', 
         transform=fig.transFigure, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7, edgecolor='navy'))

    main_grid = gridspec.GridSpec(2, 2, wspace=0.05, hspace=0.1) 
    
    # List to store captions for each scenario
    all_scenario_captions = []
    
    # Process each scenario in 2x2 grid layout
    for scenario_idx, analysis_results in enumerate(scenarios_list):
        # Determine row and column for 2x2 grid
        row, col = divmod(scenario_idx, 2)
        
        # Check if activations are non-zero (assertion only)
        for mod in modalities:
            if mod in analysis_results.streams:
                top_acts = analysis_results.get_top_activations_for_latent(mod, dim_idx)[:num_images]
                for act, _ in top_acts:
                    if act == 0:
                        return None, []
                        
        # Create subplot for this scenario - 3 rows (DINO, CLIP-img, CLIP-txt images)
        scenario_grid = gridspec.GridSpecFromSubplotSpec(3, 1, 
                                                      subplot_spec=main_grid[row, col],
                                                      height_ratios=[1, 1, 1], 
                                                      hspace=-0.1)
        
        # Add scenario label
        scenario_ax = fig.add_subplot(main_grid[row, col])
        scenario_ax.text(0.01, 1.01, f"{scenario_labels[scenario_idx]}", 
                       fontsize=10, fontweight='normal')
        scenario_ax.axis('off')
        
        # Process all three modalities for images
        for mod_idx, (mod, mod_label) in enumerate(zip(modalities, modality_labels)):
            # Get activating images based on random_sample parameter
            if random_sample:
                # Get top 50 activations first
                top_50_acts = analysis_results.get_top_activations_for_latent(mod, dim_idx)[:50]
                # Randomly sample num_images from these 50
                if len(top_50_acts) > num_images:
                    top_acts = random.sample(top_50_acts, num_images)
                else:
                    top_acts = top_50_acts[:num_images]  # In case there are fewer than num_images
            else:
                # Just get the top num_images as before
                top_acts = analysis_results.get_top_activations_for_latent(mod, dim_idx)[:num_images]
                
            # Create modality row
            ax = fig.add_subplot(scenario_grid[mod_idx])
            
            # Only show modality labels for left column (col=0)
            if col == 0:
                ax.text(-0.02, 0.5, mod_label, 
                      transform=ax.transAxes, 
                      fontsize=8, 
                      verticalalignment='center',
                      horizontalalignment='right')
            
            ax.axis('off')
            
            # Create thumbnail grid
            thumb_grid = gridspec.GridSpecFromSubplotSpec(1, num_images, 
                                                       subplot_spec=scenario_grid[mod_idx],
                                                       wspace=0.01)
            
            # Add thumbnails 
            for col_idx, (activation, img_idx) in enumerate(top_acts):
                thumb_ax = fig.add_subplot(thumb_grid[col_idx])
                img_idx = int(img_idx)
                
                # Get image from OpenImagesDataset
                img = dataset[img_idx]['image']
                
                # Process image based on center_crop parameter
                if center_crop:
                    # Center crop to square and resize
                    img = center_crop_image(img, img_size)
                else:
                    # Original behavior: just resize if possible
                    if hasattr(img, 'resize') and img_size > 0:
                        img = img.resize((img_size, img_size))
                    
                thumb_ax.imshow(img)
                
                # No activation values displayed
                thumb_ax.axis('off')
                thumb_ax.set_xticks([])
                thumb_ax.set_yticks([])
        
        # Collect captions for the current scenario
        mod = 'clip_txt'
        top_acts = analysis_results.get_top_activations_for_latent(mod, dim_idx)[:num_images]
        
        scenario_captions = []
        for activation, img_idx in top_acts:
            img_idx = int(img_idx)
            
            # Get caption from OpenImagesDataset
            sample = dataset[img_idx]
            if len(sample) > 1 and 'captions' in sample:
                caption = sample['captions']
                # If captions is a list, take the first one
                if isinstance(caption, list) and len(caption) > 0:
                    caption = caption[0]
                elif isinstance(caption, str):
                    # Caption is already a string
                    pass
                else:
                    caption = "No caption available"
            else:
                caption = "No caption available"
            
            scenario_captions.append(caption)
        
        # Add this scenario's captions to the list
        all_scenario_captions.append(scenario_captions)
    
    # Return all the captions
    return fig, all_scenario_captions


if __name__ == "__main__":

    open_images_dir = '/home/ubuntu/Projects/OpenImages/'
    checkpoint_dir = 'final_results/msae_open_global_with_cross/'
    model_path = os.path.join(checkpoint_dir, "msae_checkpoint.pth")
    run_config_path = os.path.join(checkpoint_dir, "run_config.json")

    # Load run config
    with open(run_config_path, 'r') as f:
        saved_run_config = json.load(f)
    training_args = saved_run_config['args']

    with open('configs/config_open_images.json', 'r') as f:
        config_dict = json.load(f)

    # Initialize dataset to get dimensions
    val_feature_files = config_dict['test_stream_feature_files']
    temp_dataset = HDF5FeatureDataset(stream_files=val_feature_files, return_index=False)
    d_streams = temp_dataset.get_feature_dims()
    temp_dataset.close()

    # Create and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # Get the correct model class using the SAVED training arguments
    eval_topk_type = training_args.get('topk_type', 'global') 
    SaeModelClassEval = get_sae_model_class(eval_topk_type)

    # Instantiate the correct model class with saved hyperparameters
    model = SaeModelClassEval(
        d_streams=d_streams,
        n_latents=training_args['n_latents'], 
        k=training_args['k'], 
        auxk=training_args['auxk'], 
        use_sparse_decoder=training_args['use_sparse_decoder'],
        dead_steps_threshold=training_args.get('dead_steps_threshold', 1000), 
        auxk_threshold=training_args.get('auxk_threshold', 1e-3) 
    ).to(device)

    print(f"Loading evaluation model: {SaeModelClassEval.__name__}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval();
    dataset = OpenImagesDataset(open_images_dir, 'test')
    with open('final_results/msae_open_global_with_cross/run_config.json', 'r') as f:
        config = json.load(f)



    analysis_results_global_cross = HDF5AnalysisResultsDataset('final_results/msae_open_global_with_cross/analysis_cache_val.h5', 256)
    analysis_results_global_no_cross = HDF5AnalysisResultsDataset('final_results/msae_open_global_no_cross/analysis_cache_val.h5', 256)
    analysis_results_local_cross = HDF5AnalysisResultsDataset('final_results/msae_open_local_with_cross/analysis_cache_val.h5', 256)
    analysis_results_local_no_cross = HDF5AnalysisResultsDataset('final_results/msae_open_local_no_cross/analysis_cache_val.h5', 256)

    selected_dims = list(range(2048*4))

    os.makedirs('figures_all/all_dims', exist_ok=True)

    # Load your scenarios (assuming scenarios_list is a list of analysis_results objects)
    scenarios_list = [analysis_results_local_no_cross, analysis_results_local_cross, 
                    analysis_results_global_no_cross, analysis_results_global_cross]


    for dim_idx in tqdm(selected_dims):
        if os.path.exists(f'figures_all/all_dims/{dim_idx:05d}.jpg'):
            continue
        result = create_complete_figure(
            dim_idx=dim_idx,  
            scenarios_list=scenarios_list,
            dataset=dataset,
            num_images=10,
            random_sample=False,
            center_crop=True
        )
        
        # Handle case where function returns None (zero activations)
        if result is None or result[0] is None:
            print(f"Skipping dimension {dim_idx} due to zero activations")
            continue
            
        fig, captions = result
        
        # Save figure
        save_path = f'figures_all/all_dims/{dim_idx:05d}.jpg'
        fig.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close(fig)