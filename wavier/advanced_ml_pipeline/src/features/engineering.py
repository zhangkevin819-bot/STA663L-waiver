"""
Advanced feature engineering with custom sklearn transformers.
Implements composable, stateful transformations for ML pipelines.
"""

from typing import Any, Optional

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler

from ..utils.logging import logger


class PolarsTransformer(BaseEstimator, TransformerMixin):
    """Base class for Polars-compatible sklearn transformers."""
    
    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None):
        """Fit transformer to data."""
        return self
    
    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform data."""
        return X
    
    def fit_transform(
        self,
        X: pl.DataFrame,
        y: Optional[pl.Series] = None
    ) -> pl.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class MissingValueImputer(PolarsTransformer):
    """Intelligent missing value imputation with multiple strategies."""
    
    def __init__(
        self,
        strategy: str = "median",
        fill_value: Optional[float] = None,
        add_indicator: bool = False
    ):
        self.strategy = strategy
        self.fill_value = fill_value
        self.add_indicator = add_indicator
        self.fill_values_: dict[str, float] = {}
    
    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None):
        """Learn imputation values from training data."""
        for col in X.columns:
            if X[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                if self.strategy == "mean":
                    self.fill_values_[col] = X[col].mean()
                elif self.strategy == "median":
                    self.fill_values_[col] = X[col].median()
                elif self.strategy == "mode":
                    self.fill_values_[col] = X[col].mode()[0]
                elif self.strategy == "constant":
                    self.fill_values_[col] = self.fill_value or 0.0
        
        logger.info(f"Fitted imputer with {len(self.fill_values_)} columns")
        return self
    
    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Apply imputation to data."""
        X = X.clone()
        
        for col, fill_val in self.fill_values_.items():
            if col in X.columns:
                # Add missingness indicator if requested
                if self.add_indicator:
                    X = X.with_columns(
                        pl.col(col).is_null().alias(f"{col}_was_missing")
                    )
                
                # Fill missing values
                X = X.with_columns(
                    pl.col(col).fill_null(fill_val)
                )
        
        return X


class OutlierClipper(PolarsTransformer):
    """Clip outliers using IQR or percentile methods."""
    
    def __init__(
        self,
        method: str = "iqr",
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        iqr_multiplier: float = 1.5
    ):
        self.method = method
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.iqr_multiplier = iqr_multiplier
        self.bounds_: dict[str, tuple[float, float]] = {}
    
    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None):
        """Calculate clipping bounds from training data."""
        for col in X.columns:
            if X[col].dtype in [pl.Float64, pl.Float32]:
                if self.method == "iqr":
                    q1 = X[col].quantile(0.25)
                    q3 = X[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - self.iqr_multiplier * iqr
                    upper = q3 + self.iqr_multiplier * iqr
                else:  # percentile
                    lower = X[col].quantile(self.lower_quantile)
                    upper = X[col].quantile(self.upper_quantile)
                
                self.bounds_[col] = (lower, upper)
        
        return self
    
    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Clip outliers in data."""
        X = X.clone()
        
        for col, (lower, upper) in self.bounds_.items():
            if col in X.columns:
                X = X.with_columns(
                    pl.col(col).clip(lower, upper)
                )
        
        return X


class FeatureInteractionGenerator(PolarsTransformer):
    """Generate polynomial and interaction features."""
    
    def __init__(
        self,
        columns: list[str],
        degree: int = 2,
        include_bias: bool = False
    ):
        self.columns = columns
        self.degree = degree
        self.include_bias = include_bias
    
    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Generate interaction features."""
        X = X.clone()
        
        # Generate pairwise interactions
        if self.degree >= 2:
            for i, col1 in enumerate(self.columns):
                for col2 in self.columns[i+1:]:
                    interaction_name = f"{col1}_x_{col2}"
                    X = X.with_columns(
                        (pl.col(col1) * pl.col(col2)).alias(interaction_name)
                    )
        
        # Generate polynomial features
        for col in self.columns:
            for d in range(2, self.degree + 1):
                X = X.with_columns(
                    (pl.col(col) ** d).alias(f"{col}_pow{d}")
                )
        
        logger.info(f"Generated {X.shape[1] - len(self.columns)} interaction features")
        return X


class TargetEncoder(PolarsTransformer):
    """Target-based encoding for categorical variables with smoothing."""
    
    def __init__(
        self,
        columns: list[str],
        smoothing: float = 1.0,
        min_samples: int = 1
    ):
        self.columns = columns
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.encodings_: dict[str, dict[Any, float]] = {}
        self.global_mean_: float = 0.0
    
    def fit(self, X: pl.DataFrame, y: pl.Series):
        """Learn target encodings with smoothing."""
        if y is None:
            raise ValueError("TargetEncoder requires target variable y")
        
        self.global_mean_ = float(y.mean())
        
        for col in self.columns:
            if col not in X.columns:
                continue
            
            # Calculate group statistics
            df_with_target = X.select(col).with_columns(y.alias("_target"))
            
            group_stats = df_with_target.group_by(col).agg([
                pl.col("_target").mean().alias("mean"),
                pl.col("_target").count().alias("count")
            ])
            
            # Apply smoothing
            encodings = {}
            for row in group_stats.iter_rows(named=True):
                category = row[col]
                mean = row["mean"]
                count = row["count"]
                
                # Smooth towards global mean
                smoothed = (
                    count * mean + self.smoothing * self.global_mean_
                ) / (count + self.smoothing)
                
                encodings[category] = smoothed
            
            self.encodings_[col] = encodings
        
        return self
    
    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Apply learned encodings."""
        X = X.clone()
        
        for col, encoding_map in self.encodings_.items():
            if col in X.columns:
                # Map categories to encoded values
                X = X.with_columns(
                    pl.col(col)
                    .map_dict(encoding_map, default=self.global_mean_)
                    .alias(f"{col}_encoded")
                )
        
        return X


class FeatureSelector(PolarsTransformer):
    """Select features based on importance or correlation."""
    
    def __init__(
        self,
        method: str = "variance",
        threshold: float = 0.01,
        max_features: Optional[int] = None
    ):
        self.method = method
        self.threshold = threshold
        self.max_features = max_features
        self.selected_features_: list[str] = []
    
    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None):
        """Select features based on specified method."""
        numeric_cols = [
            col for col in X.columns
            if X[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]
        
        if self.method == "variance":
            # Select features with variance above threshold
            feature_vars = {
                col: X[col].var()
                for col in numeric_cols
            }
            self.selected_features_ = [
                col for col, var in feature_vars.items()
                if var > self.threshold
            ]
        
        elif self.method == "correlation" and y is not None:
            # Select features with high correlation to target
            correlations = {
                col: abs(np.corrcoef(X[col].to_numpy(), y.to_numpy())[0, 1])
                for col in numeric_cols
            }
            sorted_features = sorted(
                correlations.items(),
                key=lambda x: x[1],
                reverse=True
            )
            self.selected_features_ = [
                col for col, _ in sorted_features[:self.max_features]
            ]
        
        logger.info(f"Selected {len(self.selected_features_)} features")
        return self
    
    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Keep only selected features."""
        return X.select(self.selected_features_)
