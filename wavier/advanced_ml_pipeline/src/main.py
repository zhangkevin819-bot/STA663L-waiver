"""
Main entry point for the ML pipeline.
Orchestrates data loading, feature engineering, training, and evaluation.
"""

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import TensorDataset, DataLoader

from src.data.loaders import CSVDataLoader, DataSplitter
from src.features.engineering import (
    MissingValueImputer,
    OutlierClipper,
    FeatureInteractionGenerator,
    FeatureSelector
)
from src.models.architectures import TransformerClassifier
from src.models.trainer import Trainer, WarmupCosineScheduler, EarlyStopping
from src.utils.config import MLConfig
from src.utils.logging import LoggerConfig, ExperimentTracker, log_execution_time, logger


def create_dataloaders(
    train_df, val_df, test_df,
    feature_cols: list[str],
    target_col: str,
    batch_size: int,
    num_workers: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders from dataframes."""
    
    def df_to_loader(df, shuffle: bool = False):
        X = torch.tensor(
            df.select(feature_cols).to_numpy(),
            dtype=torch.float32
        )
        y = torch.tensor(
            df[target_col].to_numpy(),
            dtype=torch.long
        )
        dataset = TensorDataset(X, y)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    train_loader = df_to_loader(train_df, shuffle=True)
    val_loader = df_to_loader(val_df)
    test_loader = df_to_loader(test_df)
    
    return train_loader, val_loader, test_loader


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training pipeline."""
    
    # Setup logging
    LoggerConfig.setup(
        level=cfg.logging.level,
        log_dir=Path(cfg.logging.log_dir)
    )
    
    logger.info("Starting ML Pipeline")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seeds
    torch.manual_seed(cfg.data.seed)
    
    # Initialize experiment tracker
    tracker = ExperimentTracker()
    
    with tracker.run_context("main_experiment"):
        # Step 1: Load and split data
        with log_execution_time("Data loading"):
            # In a real scenario, you'd have actual data
            # This is a placeholder for demonstration
            logger.info("Loading data...")
            
            # For demo: create synthetic data
            import numpy as np
            import polars as pl
            
            np.random.seed(cfg.data.seed)
            n_samples = 10000
            n_features = 20
            
            X = np.random.randn(n_samples, n_features)
            y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1 > 0).astype(int)
            
            # Create dataframe
            df = pl.DataFrame({
                **{f"feature_{i}": X[:, i] for i in range(n_features)},
                "target": y
            })
            
            logger.info(f"Dataset shape: {df.shape}")
        
        # Step 2: Split data
        with log_execution_time("Data splitting"):
            splitter = DataSplitter(
                train_ratio=cfg.data.train_split,
                val_ratio=cfg.data.val_split,
                test_ratio=cfg.data.test_split,
                seed=cfg.data.seed
            )
            
            train_df, val_df, test_df = splitter.split(df, "target")
        
        # Step 3: Feature engineering
        with log_execution_time("Feature engineering"):
            feature_cols = [col for col in df.columns if col != "target"]
            
            # Apply transformations
            imputer = MissingValueImputer(strategy="median")
            train_df = imputer.fit_transform(train_df)
            val_df = imputer.transform(val_df)
            test_df = imputer.transform(test_df)
            
            clipper = OutlierClipper(method="iqr")
            train_df = clipper.fit_transform(train_df)
            val_df = clipper.transform(val_df)
            test_df = clipper.transform(test_df)
            
            logger.info(f"Processed {len(feature_cols)} features")
        
        # Step 4: Create data loaders
        with log_execution_time("Creating data loaders"):
            train_loader, val_loader, test_loader = create_dataloaders(
                train_df, val_df, test_df,
                feature_cols, "target",
                cfg.data.batch_size,
                cfg.data.num_workers
            )
        
        # Step 5: Initialize model
        with log_execution_time("Model initialization"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            model = TransformerClassifier(
                input_dim=len(feature_cols),
                hidden_dim=cfg.model.hidden_dim,
                num_layers=cfg.model.num_layers,
                num_heads=cfg.model.num_heads,
                dropout=cfg.model.dropout
            )
            
            # Compile model for faster training (PyTorch 2.0+)
            if cfg.training.compile_model and hasattr(torch, 'compile'):
                logger.info("Compiling model with torch.compile")
                model = torch.compile(model)
            
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model has {total_params:,} parameters")
        
        # Step 6: Setup training
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay
        )
        
        total_steps = len(train_loader) * cfg.training.epochs
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=cfg.training.warmup_steps,
            total_steps=total_steps
        )
        
        criterion = torch.nn.CrossEntropyLoss()
        
        early_stopping = EarlyStopping(
            patience=cfg.training.early_stopping_patience
        )
        
        # Step 7: Train model
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            mixed_precision=cfg.training.mixed_precision,
            gradient_clip=cfg.training.gradient_clip,
            accumulation_steps=cfg.training.gradient_accumulation_steps
        )
        
        with log_execution_time("Model training"):
            history = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=cfg.training.epochs,
                scheduler=scheduler,
                early_stopping=early_stopping,
                checkpoint_dir=Path("checkpoints")
            )
        
        # Step 8: Final evaluation
        with log_execution_time("Final evaluation"):
            test_metrics = trainer.validate(test_loader)
            logger.info(f"Test metrics: {test_metrics}")
        
        logger.info("Pipeline completed successfully!")
        
        return test_metrics


if __name__ == "__main__":
    main()
