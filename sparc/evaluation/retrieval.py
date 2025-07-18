import torch
import os
import json
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from typing import Dict, Tuple, Optional, List, TYPE_CHECKING
import torch.nn.functional as F
from sparc.post_analysis import HDF5AnalysisResultsDataset

def eval_retrieval(analysis_results: HDF5AnalysisResultsDataset, args, config_dict, wandb_run=None):
    if args.verbose: print("--- Starting Retrieval Evaluation ---")

    if not isinstance(analysis_results, HDF5AnalysisResultsDataset):
         raise TypeError("analysis_results must be an instance of HDF5AnalysisResultsDataset.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.verbose: print(f"Using device: {device} for retrieval computations")

    available_streams = analysis_results.streams
    if not available_streams:
        print("Error: No streams found in HDF5AnalysisResultsDataset for retrieval.")
        return None

    if args.verbose: print(f"Found {len(available_streams)} streams in HDF5 analysis results.")

    # --- Retrieval Metrics Evaluation ---
    all_retrieval_metrics = defaultdict(dict)
    if args.verbose: print("Evaluating retrieval metrics iteratively...")

    num_streams = len(available_streams)
    cross_recon_keys = analysis_results.get_cross_reconstruction_keys()
    num_cross_recon_tasks = len(cross_recon_keys)
    # Total tasks: SelfRecon (N) + LatentAlignment (N*(N-1)) + CrossRecon (num_cross_recon_keys)
    total_retrieval_tasks = num_streams + num_streams * (num_streams - 1) + num_cross_recon_tasks
    pbar = tqdm(total=total_retrieval_tasks, desc="Evaluating Retrieval", disable=not args.verbose, position=0, leave=True)

    # 1. Self-Reconstruction Recall (Recon vs Raw)
    for stream in available_streams:
        key = f"{stream.upper()}_from_{stream.upper()}"
        if args.verbose: pbar.set_description(f"Eval SelfRecon {key}")

        recon_array = analysis_results.get_all_features_for_stream(stream, 'recon')
        raw_array = analysis_results.get_all_features_for_stream(stream, 'raw')

        if recon_array is None or raw_array is None or recon_array.size == 0 or raw_array.size == 0:
            pbar.update(1)
            raise ValueError(f"Missing or empty 'recon' or 'raw' data for stream '{stream}'. Cannot compute SelfRecon.")

        recon_tensor = torch.from_numpy(recon_array).float()
        raw_tensor = torch.from_numpy(raw_array).float()
        metric = run_downstream_retrieval_eval(
            query_feats=recon_tensor, 
            reference_feats=raw_tensor, 
            verbose=args.verbose, 
            task_desc=f"SelfRecon {key}",
            device=str(device)
        )
        all_retrieval_metrics['SelfRecon'][key] = metric
        pbar.update(1)

    # 2. Latent Space Alignment (Latent vs Latent)
    for src_stream in available_streams:
        for tgt_stream in available_streams:
            if src_stream == tgt_stream: continue
            key = f"{tgt_stream.upper()}_from_{src_stream.upper()}"
            if args.verbose: pbar.set_description(f"Eval LatentAlign {key}")

            src_latent_array = analysis_results.get_all_features_for_stream(src_stream, 'latents')
            tgt_latent_array = analysis_results.get_all_features_for_stream(tgt_stream, 'latents')

            if src_latent_array is None or tgt_latent_array is None or src_latent_array.size == 0 or tgt_latent_array.size == 0:
                pbar.update(1)
                raise ValueError(f"Missing or empty 'latents' data for source '{src_stream}' or target '{tgt_stream}'. Cannot compute LatentAlignment.")

            src_latent = torch.from_numpy(src_latent_array).float()
            tgt_latent = torch.from_numpy(tgt_latent_array).float()
            metric = run_downstream_retrieval_eval(
                query_feats=src_latent, 
                reference_feats=tgt_latent, 
                verbose=args.verbose, 
                task_desc=f"LatentAlign {key}",
                device=str(device)
            )
            all_retrieval_metrics['LatentAlignment'][key] = metric
            pbar.update(1)

    # 3. Cross-Reconstruction Recall (Cross-Recon vs Target Raw)
    for cross_key in cross_recon_keys:
        parts = cross_key.split('_from_')
        if len(parts) != 2:
            pbar.update(1)
            print(f"Warning: Could not parse cross-reconstruction key '{cross_key}'. Skipping.")
            continue

        tgt_stream_cr, src_stream_cr = parts
        metric_key = f"{tgt_stream_cr.upper()}_from_{src_stream_cr.upper()}"
        if args.verbose: pbar.set_description(f"Eval CrossRecon {metric_key}")

        cross_recon_array = analysis_results.get_all_cross_reconstruction_features(cross_key)
        raw_target_array = analysis_results.get_all_features_for_stream(tgt_stream_cr, 'raw')

        if cross_recon_array is None or raw_target_array is None or cross_recon_array.size == 0 or raw_target_array.size == 0:
            pbar.update(1)
            raise ValueError(f"Missing or empty cross-reconstruction data for '{cross_key}' or raw data for target '{tgt_stream_cr}'. Cannot compute CrossReconRecall.")

        cross_recon_tensor = torch.from_numpy(cross_recon_array).float()
        raw_target_tensor = torch.from_numpy(raw_target_array).float()
        metric = run_downstream_retrieval_eval(
            query_feats=cross_recon_tensor, 
            reference_feats=raw_target_tensor, 
            verbose=args.verbose, 
            task_desc=f"CrossRecon {metric_key}",
            device=str(device)
        )
        all_retrieval_metrics['CrossReconRecall'][metric_key] = metric
        pbar.update(1)

    pbar.close()

    # --- Results Saving and Logging ---
    results_path = os.path.join(args.checkpoint_dir, "retrieval_results.json")
    if args.verbose: print(f"Saving retrieval results to: {results_path}")
    # Convert tensors to lists for JSON serialization
    serializable_metrics = {}
    for category, metrics_dict in all_retrieval_metrics.items():
        serializable_metrics[category] = {}
        for key, metric_val in metrics_dict.items():
            serializable_metrics[category][key] = {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in metric_val.items()
            }

    with open(results_path, 'w') as f: json.dump(serializable_metrics, f, indent=4)

    all_r1_scores = []
    for category, metrics_dict in serializable_metrics.items():
         for key, metric_val in metrics_dict.items():
              if 'R@1' in metric_val:
                   all_r1_scores.append(metric_val['R@1'])

    final_avg_R1 = np.mean(all_r1_scores) if all_r1_scores else 0.0
    if args.verbose: print(f"Calculated final average R@1: {final_avg_R1:.4f}")

    if wandb_run and serializable_metrics:
        if args.verbose: print("Logging retrieval summary to Wandb...")
        try:
            wandb_retrieval_log = {}
            for category, metrics_dict in serializable_metrics.items():
                 for key, metric_val in metrics_dict.items():
                     for r_k in ['R@1', 'R@5', 'R@10']:
                         if r_k in metric_val:
                             wandb_key = f"retrieval/{category}/{key}/{r_k}"
                             wandb_retrieval_log[wandb_key] = metric_val[r_k]

            if wandb_retrieval_log:
                wandb_retrieval_log["retrieval/final_avg_R1"] = final_avg_R1
                wandb_run.log(wandb_retrieval_log)
        except Exception as e:
             print(f"Warning: Failed to log retrieval summary to Wandb: {e}")
        
    if args.verbose: print("--- Retrieval Evaluation Finished ---")
    
    return final_avg_R1


# --- Retrieval Evaluation Function ---
@torch.no_grad()
# This function should be updated. The main reason is we don't need batching for retrieval when latents are sparse. Sparse dot product is fast.
# Need to use scipy.sparse.csr_matrix for the similarity matrix.
def run_downstream_retrieval_eval(
    query_feats: np.ndarray | torch.Tensor,
    reference_feats: np.ndarray | torch.Tensor,
    k_list: Tuple[int, ...] = (1, 5, 10),
    return_ranks: bool = False,
    device: Optional[str] = None,
    batch_size: int = 512,
    verbose: bool = False,
    task_desc: str = "Retrieval Batch"
) -> Dict[str, float | torch.Tensor]:
    """
    Evaluates retrieval performance between query and reference feature sets using batch processing.

    Computes Recall@k, Median Rank (MedR), and Mean Reciprocal Rank (MRR)
    based on the rank of the ground-truth reference for each query.
    Ground truth assumes query_feats[i] corresponds to reference_feats[i].
    Uses batching to handle large datasets without OOM errors from the similarity matrix.

    Args:
        query_feats (np.ndarray | torch.Tensor): Query features (N, D). Will be L2 normalized.
        reference_feats (np.ndarray | torch.Tensor): Reference features (the set to search within) (N, D). Will be L2 normalized.
        k_list (Tuple[int, ...]): Tuple of k values for Recall@k. Defaults to (1, 5, 10).
        return_ranks (bool): If True, includes the raw ranks tensor in the
                             output dictionary under the key 'ranks'. Defaults to False.
        device (Optional[str]): Device for computation ('cpu', 'cuda:0', etc.). Defaults to query features' device.
        batch_size (int): Number of queries to process at a time to manage memory. Defaults to 512.
        verbose (bool): Whether to show the inner progress bar for batch processing.
        task_desc (str): Description for the inner progress bar.

    Returns:
        Dict[str, float | torch.Tensor]: Dictionary containing R@k for each k in k_list,
                                         MedR, MRR, and optionally 'ranks'.
                                         Metrics are floats, ranks is a torch.Tensor.
    """

    # 1. Convert to tensors and validate shapes
    query = torch.as_tensor(query_feats, dtype=torch.float32)
    reference = torch.as_tensor(reference_feats, dtype=torch.float32)
    N, D = query.shape

    if reference.shape[0] != N or reference.shape[1] != D:
         raise ValueError(f"Query and reference features must have the same shape (N, D). Got query: {query.shape}, reference: {reference.shape}")
    if query.ndim != 2:
        raise ValueError(f"Features must be 2D tensors (N, D). Got query: {query.shape}")

    if device is not None:
        processing_device = torch.device(device)
        query, reference = query.to(processing_device), reference.to(processing_device)
    else:
        processing_device = query.device

    # 2. L2-normalize reference features (do this once)
    reference_norm = F.normalize(reference, p=2, dim=-1, eps=1e-8)

    # 3. Process queries in batches to compute ranks
    all_ranks = []
    # Inner progress bar for batch processing within this specific task
    inner_pbar = tqdm(total=N, desc=task_desc, leave=False, disable=not verbose, unit='samples', position=1)
    for i in range(0, N, batch_size):
        query_batch = query[i : i + batch_size]
        current_batch_size = query_batch.shape[0]

        query_batch_norm = F.normalize(query_batch, p=2, dim=-1, eps=1e-8)

        # Similarity matrix for the batch (batch_size x N)
        sim_batch = query_batch_norm @ reference_norm.T

        sorted_idx_batch = torch.argsort(sim_batch, dim=1, descending=True)

        # Ground truth targets for this batch (original indices)
        target_batch = torch.arange(i, i + current_batch_size, device=processing_device)

        # Find the rank of the true target for each query in the batch
        # Compare sorted indices with the target index broadcasted across columns
        ranks_batch = (sorted_idx_batch == target_batch[:, None]).nonzero(as_tuple=False)[:, 1] + 1

        all_ranks.append(ranks_batch.cpu())
        inner_pbar.update(current_batch_size)
    
    inner_pbar.close()

    ranks = torch.cat(all_ranks)

    # 4. Calculate metrics from the collected ranks
    metrics: Dict[str, float | torch.Tensor] = {
        f"R@{k}": (ranks <= k).float().mean().item() for k in k_list
    }
    metrics["MedR"] = ranks.median().item()
    metrics["MRR"]  = (1.0 / ranks.float()).mean().item()

    if return_ranks:
        metrics["ranks"] = ranks

    return metrics