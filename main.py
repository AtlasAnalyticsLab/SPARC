import torch
from torch.utils.data import DataLoader
from sparc.datasets import HDF5FeatureDataset
from sparc.model import get_sae_model_class
from sparc.post_analysis import get_analysis_results_h5
import numpy as np
import os
import torch.optim as optim
from sparc.train import train_loop
import argparse
import json
from sparc.evaluation.retrieval import eval_retrieval
from sparc.evaluation.classification import eval_classification
from sparc.utils import seed_worker, set_seed
import wandb
from torch.utils.data import Sampler
import random

class ContiguousRandomBatchSampler(Sampler[list[int]]):
    """
    Yields lists of contiguous indices of length `batch_size`,
    shuffled at the batch level each epoch.
    """
    def __init__(self, n_samples: int, batch_size: int, drop_last: bool = False):
        self.n     = n_samples
        self.bs    = batch_size
        self.nb    = n_samples // batch_size
        self.rem   = n_samples % batch_size
        self.drop  = drop_last

    def __iter__(self):
        # random order of whole batches
        for b in torch.randperm(self.nb):
            start = b * self.bs
            yield list(range(start, start + self.bs))

        # optional last (smaller) batch
        if not self.drop and self.rem:
            tail = list(range(self.nb * self.bs, self.n))
            random.shuffle(tail)         
            yield tail

    def __len__(self):
        return self.nb + (0 if self.drop or self.rem == 0 else 1)

def train(args, config_dict, wandb_run=None):
    """
    Trains the SPARC model using features loaded from separate files.
    Returns the trained model.
    """

    # Seed for reproducibility
    set_seed(args.seed)
    split_generator = torch.Generator()
    split_generator.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # training feature files
    train_feature_files = config_dict.get('train_stream_feature_files')
    if not train_feature_files or not isinstance(train_feature_files, dict):
        raise ValueError("'train_stream_feature_files' missing or invalid in config file.")
        
    # validation feature files
    val_feature_files = config_dict.get('val_stream_feature_files')
    has_validation = val_feature_files and isinstance(val_feature_files, dict)
        
    if args.verbose:
        print(f"Using device: {device}")
        print(f"Loading training features from {len(train_feature_files)} files specified in config:")
        for stream_name, file_path in train_feature_files.items():
             print(f"  {stream_name.upper()}: {file_path}")
        if has_validation:
            print(f"Loading validation features from {len(val_feature_files)} files specified in config:")
            for stream_name, file_path in val_feature_files.items():
                print(f"  {stream_name.upper()}: {file_path}")

    # --- Save configuration --- 
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Save the *run* configuration (args + config_dict)
    run_config_to_save = {
        "args": vars(args),
        "config_file_content": config_dict 
    }
    config_save_path = os.path.join(args.checkpoint_dir, "run_config.json")
    with open(config_save_path, 'w') as f:
        # Use default=str to handle potential non-serializable types in args (like Namespace)
        json.dump(run_config_to_save, f, indent=4, default=str) 
    if args.verbose:
        print(f"Combined run configuration saved to: {config_save_path}")
    # ------------------------

    # config_dict contains the stream_files
    dataset = HDF5FeatureDataset(
        stream_files=train_feature_files,
    )
    
    
    """
    Shuffle=True makes the H5 load significantly slower compared to not shuffling. 
    Difference is hours vs minutes per epcoh for ~1M train samples.
    Hence we use a custom batch sampler that shuffles the batches at the batch level instead of the sample level.
    In our experiments the loss didn't change that much by using the custom batch sampler but wall clock time is reduced by orders of magnitude.
    You can uncomment the dataloader below to use the default shuffle.
    """

    # dataloader = DataLoader(
    #     dataset, batch_size=args.batch_size, shuffle=True,
    #     num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=split_generator
    # )
    batch_sampler = ContiguousRandomBatchSampler(len(dataset), args.batch_size)
    dataloader = DataLoader(
        dataset, batch_sampler=batch_sampler,
        num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=split_generator
    )
    
    # Create validation dataloader if validation files are provided
    val_dataloader = None
    if has_validation:
        val_dataset = HDF5FeatureDataset(
            stream_files=val_feature_files,
        )
        val_batch_sampler = ContiguousRandomBatchSampler(len(val_dataset), args.batch_size)
        val_dataloader = DataLoader(
            val_dataset, batch_sampler=val_batch_sampler,
            num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=split_generator
        )
        if args.verbose:
            print(f"Created validation dataloader with {len(val_dataset)} samples")
    
    # Get feature dimensions from the dataset
    d_streams = dataset.get_feature_dims()
    if args.verbose:
        print("Detected feature dimensions:", d_streams)

    # --- Initialize Model ---
    # Get the appropriate model class based on the argument
    SaeModelClass = get_sae_model_class(args.topk_type)
    
    # Instantiate the selected model class. With global topk, it will the SPARC model.
    model = SaeModelClass(
        d_streams=d_streams,
        n_latents=args.n_latents, 
        k=args.k, 
        auxk=args.auxk, 
        use_sparse_decoder=args.use_sparse_decoder,
        dead_steps_threshold=args.dead_steps_threshold, 
        auxk_threshold=args.auxk_threshold 
    ).to(device)

    print(f"Initialized {SaeModelClass.__name__} with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-8)

    # Train model using the train_loop 
    if args.verbose: print("Starting multi-stream training...")
    model, metrics = train_loop(
        model=model,
        dataloader=dataloader, 
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        dead_neuron_threshold=args.dead_neuron_threshold,
        auxk_coef=args.auxk_coef,
        cross_loss_coef=args.cross_loss_coef,
        wandb_run=wandb_run,
        val_dataloader=val_dataloader
    )

    if args.verbose: print(f"Saving model and metrics to {args.checkpoint_dir}...")
    model_save_path = os.path.join(args.checkpoint_dir, "msae_checkpoint.pth")
    torch.save(model.state_dict(), model_save_path)
    if args.verbose: print(f"  Model saved to: {model_save_path}")

    # Save metrics
    metrics_save_path = os.path.join(args.checkpoint_dir, "training_metrics.json")
    serializable_metrics = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) 
        for k, v in metrics.items()
    }
    with open(metrics_save_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    if args.verbose: print(f"  Metrics saved to: {metrics_save_path}")

    dataset.close()
    if has_validation:
        val_dataset.close()
    if args.verbose: print("Training complete!")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate a Multi-Stream Sparse Autoencoder.")

    # --- Config File Argument --- 
    parser.add_argument("--config", type=str, default="configs/config_coco.json", 
                        help="Path to JSON config file containing dataset paths.")

    # --- Model Hyperparameters --- (Used for Training)
    parser.add_argument("--n_latents", type=int, default=8192, help="Number of latent dimensions in the SAE.")
    parser.add_argument("--k", type=int, default=64, help="Sparsity parameter (number of active latents).")
    parser.add_argument("--auxk", type=int, default=64, help="AuxK parameter (number of latents for auxiliary loss).")
    parser.add_argument("--use_sparse_decoder", action='store_true', help="Use sparse decoder kernel (requires Triton).")

    # --- Training Hyperparameters ---
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam optimizer.")
    parser.add_argument("--dead_neuron_threshold", type=int, default=1000, help="Steps threshold to consider a neuron dead and reinitialize.")
    parser.add_argument("--auxk_coef", type=float, default=1/32, help="Coefficient for the AuxK loss term.")
    parser.add_argument("--cross_loss_coef", type=float, default=1.0, help="Coefficient for the cross-stream reconstruction loss.")
    
    # --- Training/Evaluation Control ---
    parser.add_argument("--only_train", action='store_true', help="Only run training and skip the evaluation phase.")
    parser.add_argument("--only_eval", action='store_true', help="Only run evaluation and skip the training phase.")
    
    # --- Evaluation Parameters (Required if evaluation runs) ---
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training ratio for downstream classifier (Required for eval).")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for downstream classifier (Required for eval).")

    # --- Experiment Setup & Other Args ---
    parser.add_argument("--exp_name", type=str, default="sparc_coco", help="Experiment name, used for subfolder in output_dir.")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for training/analysis results.") 
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output during training/analysis.")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name.")
    
    # --- New arguments for get_sae_model_class ---
    parser.add_argument("--topk_type", type=str, default='global', choices=['local', 'global'], help="Type of top-k similarity calculation.")
    parser.add_argument("--dead_steps_threshold", type=int, default=1000, help="Steps threshold to consider a neuron dead and reinitialize.")
    parser.add_argument("--auxk_threshold", type=float, default=1e-3, help="Activation threshold above which a selected neuron is considered active enough to reset its dead counter.")

    args = parser.parse_args()

    # --- Ensure only_train and only_eval are mutually exclusive ---
    if args.only_train and args.only_eval:
        parser.error("Arguments --only_train and --only_eval are mutually exclusive.")

    # --- Load Config Dictionary ---
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    
    if args.verbose: 
        print(f"Loaded configuration from: {args.config}")

    # --- Set Seed ---
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    # --- Create Output Directory & Add to Args --- 
    checkpoint_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True) 
    args.checkpoint_dir = checkpoint_dir

    # --- Wandb Initialization (Initialize once, reuse if needed) ---
    wandb_run = None
    if args.wandb_project:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.exp_name,
            config={"args": vars(args), "config": config_dict} 
        )
        if args.verbose: print(f"Initialized Wandb run for project '{args.wandb_project}', run name '{args.exp_name}'")

    # --- Train Model (if not only_eval) --- 
    model = None  
    # Initialize dataset to get dimensions
    val_feature_files = config_dict['val_stream_feature_files']
    temp_dataset = HDF5FeatureDataset(stream_files=val_feature_files, return_index=False)
    d_streams = temp_dataset.get_feature_dims()
    temp_dataset.close()
    
    if not args.only_eval:
        if args.verbose: print("--- Starting Training --- ")
        model = train(args, config_dict, wandb_run=wandb_run) 
        if args.verbose: print("--- Training Finished --- ")
    else:
        if args.verbose: print("\nSkipping training as --only_eval flag was provided.")
        # Load model for evaluation - simplified, without try-except
        if args.verbose: print("Loading model for evaluation...")
        model_path = os.path.join(args.checkpoint_dir, "msae_checkpoint.pth")
        run_config_path = os.path.join(args.checkpoint_dir, "run_config.json")
        
        # Load run config
        with open(run_config_path, 'r') as f:
            saved_run_config = json.load(f)
        training_args = saved_run_config['args']
    
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
        model.eval()

    # --- Run Evaluation (if not only_train) --- 
    classification_summary = None 
    final_avg_R1 = None 
    if not args.only_train: 
        if args.verbose: print("\n--- Starting Evaluation --- ")
        analysis_results = get_analysis_results_h5(args.checkpoint_dir, config_dict, model, 
                                                   args.batch_size, args.seed, verbose=True,
                                                   cache_filename = "analysis_cache_val.h5")
        # # print("Running classification evaluation...")
        print("Running classification evaluation...")
        classification_summary = eval_classification(analysis_results, args, config_dict, 
                                                     wandb_run=wandb_run, verbose=args.verbose)

        # Evaluating Retrieval for global model is done in local setting to avoid data leakage. 
        # Otherwise, would get 0.99 R@1 for latents using SPARC (with global topk).
        analysis_results_local = analysis_results

        run_config_path = os.path.join(args.checkpoint_dir, "run_config.json")
        # Load run config
        with open(run_config_path, 'r') as f:
            saved_run_config = json.load(f)
        training_args = saved_run_config['args']
        eval_topk_type = training_args.get('topk_type', 'global')
        if eval_topk_type == 'global':
            print("Computing post analysis for global model in local setting to avoid data leakage.")
            SaeModelClassEval = get_sae_model_class('local')
            model = SaeModelClassEval(
                d_streams=d_streams,
                n_latents=training_args['n_latents'], 
                k=training_args['k'], 
                auxk=training_args['auxk'], 
                use_sparse_decoder=training_args['use_sparse_decoder'],
                dead_steps_threshold=training_args.get('dead_steps_threshold', 1000), 
                auxk_threshold=training_args.get('auxk_threshold', 1e-3) 
            ).to(device)
            model_path = os.path.join(args.checkpoint_dir, "msae_checkpoint.pth")

            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()
            analysis_results_local = get_analysis_results_h5(args.checkpoint_dir, config_dict, model, 
                                                             args.batch_size, args.seed, verbose=True,
                                                             cache_filename = "analysis_cache_retrieval_val.h5")
        print("Running retrieval evaluation...")
        final_avg_R1 = eval_retrieval(analysis_results_local, args, config_dict, wandb_run=wandb_run)
        
        if args.verbose: print("--- Evaluation Finished --- ")
        analysis_results_local.close()
        if eval_topk_type == 'global':
            analysis_results.close()
    else:
        if args.verbose: print("\nSkipping evaluation as --only_train flag was provided.")

            
    # --- Finish Wandb Run (if initialized) ---
    if wandb_run is not None:
        if args.verbose: print("Finishing Wandb run...")
        wandb.finish()
