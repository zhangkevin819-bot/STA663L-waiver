"""
Configuration management using Hydra with type-safe dataclasses.
Implements hierarchical configs with runtime validation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class DataConfig:
    """Data processing configuration."""
    raw_path: Path = Path("data/raw")
    processed_path: Path = Path("data/processed")
    batch_size: int = 256
    num_workers: int = 4
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42
    use_polars: bool = True
    cache_enabled: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str = "transformer_classifier"
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    activation: str = "gelu"
    use_flash_attention: bool = True
    pretrained: Optional[str] = None


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine_with_warmup"
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    compile_model: bool = True
    early_stopping_patience: int = 10
    gradient_accumulation_steps: int = 1


@dataclass
class InferenceConfig:
    """Inference and serving configuration."""
    device: str = "cuda"
    batch_size: int = 64
    num_workers: int = 2
    enable_onnx: bool = False
    quantization: Optional[str] = None
    optimization_level: int = 2


@dataclass
class LoggingConfig:
    """Logging and experiment tracking."""
    level: str = "INFO"
    log_dir: Path = Path("logs")
    mlflow_uri: str = "http://localhost:5000"
    experiment_name: str = "advanced_ml_pipeline"
    track_gradients: bool = True
    log_every_n_steps: int = 50


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    log_level: str = "info"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    rate_limit: int = 100
    timeout: float = 30.0


@dataclass
class MLConfig:
    """Root configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    # Global settings
    project_root: Path = Path.cwd()
    environment: str = "development"
    debug: bool = False


def register_configs() -> None:
    """Register all configuration schemas with Hydra."""
    cs = ConfigStore.instance()
    cs.store(name="config", node=MLConfig)
    cs.store(group="data", name="base_data", node=DataConfig)
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="training", name="base_training", node=TrainingConfig)


# Auto-register configs on module import
register_configs()
