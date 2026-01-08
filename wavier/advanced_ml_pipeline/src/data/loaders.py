"""
High-performance data loading with Polars and custom transforms.
Implements efficient data pipelines with caching and validation.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from ..utils.logging import log_execution_time, logger


class DataValidator(Protocol):
    """Protocol for data validation strategies."""
    
    def validate(self, df: pl.DataFrame) -> tuple[bool, Optional[str]]:
        """Validate dataframe, return (is_valid, error_message)."""
        ...


class DataLoader(ABC):
    """Abstract base class for data loaders."""
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        validate: bool = True
    ):
        self.cache_dir = cache_dir
        self.validate = validate
        
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def load_raw(self) -> pl.DataFrame:
        """Load raw data from source."""
        pass
    
    @abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply transformations to raw data."""
        pass
    
    def load(self) -> pl.DataFrame:
        """Load and transform data with caching."""
        cache_path = self.cache_dir / "processed.parquet" if self.cache_dir else None
        
        if cache_path and cache_path.exists():
            logger.info(f"Loading cached data from {cache_path}")
            return pl.read_parquet(cache_path)
        
        with log_execution_time("Data loading and transformation"):
            df = self.load_raw()
            df = self.transform(df)
            
            if self.validate:
                self._validate_data(df)
            
            if cache_path:
                df.write_parquet(cache_path)
                logger.info(f"Cached processed data to {cache_path}")
        
        return df
    
    def _validate_data(self, df: pl.DataFrame) -> None:
        """Validate processed data."""
        if df.is_empty():
            raise ValueError("Processed dataframe is empty")
        
        null_counts = df.null_count()
        if null_counts.sum_horizontal()[0] > 0:
            logger.warning(f"Found null values: {null_counts}")


class CSVDataLoader(DataLoader):
    """Efficient CSV data loader using Polars."""
    
    def __init__(
        self,
        file_path: Path,
        target_column: str,
        feature_columns: Optional[list[str]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.file_path = file_path
        self.target_column = target_column
        self.feature_columns = feature_columns
    
    def load_raw(self) -> pl.DataFrame:
        """Load CSV with optimized reading."""
        logger.info(f"Reading CSV from {self.file_path}")
        
        return pl.read_csv(
            self.file_path,
            infer_schema_length=10000,
            try_parse_dates=True,
        )
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply standard transformations."""
        # Select columns if specified
        if self.feature_columns:
            columns = self.feature_columns + [self.target_column]
            df = df.select(columns)
        
        # Drop rows with nulls in target
        df = df.filter(pl.col(self.target_column).is_not_null())
        
        # Optimize dtypes
        df = self._optimize_dtypes(df)
        
        return df
    
    @staticmethod
    def _optimize_dtypes(df: pl.DataFrame) -> pl.DataFrame:
        """Optimize column dtypes for memory efficiency."""
        optimizations = []
        
        for col in df.columns:
            dtype = df[col].dtype
            
            # Downcast integers
            if dtype in [pl.Int64, pl.Int32]:
                min_val = df[col].min()
                max_val = df[col].max()
                
                if min_val >= 0:
                    if max_val < 256:
                        optimizations.append(pl.col(col).cast(pl.UInt8))
                    elif max_val < 65536:
                        optimizations.append(pl.col(col).cast(pl.UInt16))
                else:
                    if -128 <= min_val and max_val < 127:
                        optimizations.append(pl.col(col).cast(pl.Int8))
                    elif -32768 <= min_val and max_val < 32767:
                        optimizations.append(pl.col(col).cast(pl.Int16))
        
        if optimizations:
            df = df.with_columns(optimizations)
        
        return df


class DataSplitter:
    """Handles train/val/test splitting with stratification."""
    
    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        stratify: bool = True
    ):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.stratify = stratify
    
    def split(
        self,
        df: pl.DataFrame,
        target_col: str
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Split dataframe into train/val/test sets."""
        logger.info(
            f"Splitting data: train={self.train_ratio:.1%}, "
            f"val={self.val_ratio:.1%}, test={self.test_ratio:.1%}"
        )
        
        # Convert to numpy for sklearn splitting
        indices = np.arange(len(df))
        y = df[target_col].to_numpy() if self.stratify else None
        
        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_ratio,
            stratify=y if y is not None else None,
            random_state=self.seed
        )
        
        # Second split: train vs val
        val_size = self.val_ratio / (self.train_ratio + self.val_ratio)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            stratify=y[train_val_idx] if y is not None else None,
            random_state=self.seed
        )
        
        # Create splits
        train_df = df[train_idx]
        val_df = df[val_idx]
        test_df = df[test_idx]
        
        logger.info(
            f"Split sizes: train={len(train_df)}, "
            f"val={len(val_df)}, test={len(test_df)}"
        )
        
        return train_df, val_df, test_df


class DataAugmenter:
    """Composable data augmentation pipeline."""
    
    def __init__(self, transforms: list[Callable[[pl.DataFrame], pl.DataFrame]]):
        self.transforms = transforms
    
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply augmentation pipeline."""
        for transform in self.transforms:
            df = transform(df)
        return df
    
    @staticmethod
    def add_noise(
        columns: list[str],
        noise_level: float = 0.01
    ) -> Callable[[pl.DataFrame], pl.DataFrame]:
        """Add Gaussian noise to numeric columns."""
        def _transform(df: pl.DataFrame) -> pl.DataFrame:
            for col in columns:
                if col in df.columns:
                    std = df[col].std()
                    noise = np.random.normal(0, std * noise_level, len(df))
                    df = df.with_columns(
                        (pl.col(col) + noise).alias(col)
                    )
            return df
        return _transform
