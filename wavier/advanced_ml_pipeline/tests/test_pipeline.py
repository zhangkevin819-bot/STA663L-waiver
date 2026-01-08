"""
Comprehensive test suite using pytest with fixtures and property-based testing.
"""

import pytest
import torch
import numpy as np
import polars as pl
from hypothesis import given, strategies as st

from src.data.loaders import DataSplitter
from src.features.engineering import (
    MissingValueImputer,
    OutlierClipper,
    FeatureInteractionGenerator
)
from src.models.architectures import TransformerClassifier, MultiHeadAttention


# Fixtures
@pytest.fixture
def sample_dataframe():
    """Create a sample Polars dataframe for testing."""
    np.random.seed(42)
    return pl.DataFrame({
        'feature_1': np.random.randn(1000),
        'feature_2': np.random.randn(1000),
        'feature_3': np.random.randn(1000),
        'target': np.random.randint(0, 2, 1000)
    })


@pytest.fixture
def sample_tensor():
    """Create a sample PyTorch tensor for testing."""
    return torch.randn(32, 10, 512)


# Data Tests
class TestDataSplitter:
    """Tests for DataSplitter class."""
    
    def test_split_ratios(self, sample_dataframe):
        """Test that split ratios are correct."""
        splitter = DataSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        train, val, test = splitter.split(sample_dataframe, 'target')
        
        total = len(train) + len(val) + len(test)
        assert abs(len(train) / total - 0.8) < 0.05
        assert abs(len(val) / total - 0.1) < 0.05
        assert abs(len(test) / total - 0.1) < 0.05
    
    def test_no_data_leakage(self, sample_dataframe):
        """Test that splits don't overlap."""
        splitter = DataSplitter()
        train, val, test = splitter.split(sample_dataframe, 'target')
        
        # Convert to sets of row hashes
        train_rows = set(train.rows())
        val_rows = set(val.rows())
        test_rows = set(test.rows())
        
        assert len(train_rows & val_rows) == 0
        assert len(train_rows & test_rows) == 0
        assert len(val_rows & test_rows) == 0
    
    def test_reproducibility(self, sample_dataframe):
        """Test that splits are reproducible with same seed."""
        splitter1 = DataSplitter(seed=42)
        train1, val1, test1 = splitter1.split(sample_dataframe, 'target')
        
        splitter2 = DataSplitter(seed=42)
        train2, val2, test2 = splitter2.split(sample_dataframe, 'target')
        
        assert train1.equals(train2)
        assert val1.equals(val2)
        assert test1.equals(test2)


# Feature Engineering Tests
class TestMissingValueImputer:
    """Tests for MissingValueImputer."""
    
    def test_median_imputation(self):
        """Test median imputation strategy."""
        df = pl.DataFrame({
            'a': [1.0, 2.0, None, 4.0],
            'b': [None, 2.0, 3.0, 4.0]
        })
        
        imputer = MissingValueImputer(strategy='median')
        result = imputer.fit_transform(df)
        
        assert result['a'].null_count() == 0
        assert result['b'].null_count() == 0
    
    def test_imputer_preserves_non_null(self):
        """Test that imputer doesn't modify non-null values."""
        df = pl.DataFrame({'a': [1.0, 2.0, 3.0]})
        
        imputer = MissingValueImputer()
        result = imputer.fit_transform(df)
        
        assert result['a'].to_list() == [1.0, 2.0, 3.0]


class TestOutlierClipper:
    """Tests for OutlierClipper."""
    
    def test_iqr_clipping(self):
        """Test IQR-based outlier clipping."""
        df = pl.DataFrame({
            'a': [1.0, 2.0, 3.0, 100.0]  # 100 is outlier
        })
        
        clipper = OutlierClipper(method='iqr')
        result = clipper.fit_transform(df)
        
        assert result['a'].max() < 100.0


# Model Tests
class TestTransformerClassifier:
    """Tests for TransformerClassifier."""
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = TransformerClassifier(
            input_dim=10,
            hidden_dim=64,
            num_layers=2,
            num_classes=2
        )
        
        x = torch.randn(8, 10)
        output = model(x)
        
        assert output.shape == (8, 2)
    
    def test_sequential_input(self):
        """Test model handles sequential input."""
        model = TransformerClassifier(
            input_dim=10,
            hidden_dim=64,
            num_layers=2
        )
        
        x = torch.randn(8, 20, 10)  # (batch, seq, features)
        output = model(x)
        
        assert output.shape == (8, 2)
    
    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 10),
        (4, 20),
        (16, 50)
    ])
    def test_various_batch_sizes(self, batch_size, seq_len):
        """Test model with various batch sizes."""
        model = TransformerClassifier(input_dim=10, hidden_dim=64, num_layers=2)
        x = torch.randn(batch_size, seq_len, 10)
        output = model(x)
        
        assert output.shape == (batch_size, 2)


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""
    
    def test_attention_output_shape(self):
        """Test attention produces correct output shape."""
        attn = MultiHeadAttention(dim=512, num_heads=8)
        x = torch.randn(2, 10, 512)
        output = attn(x)
        
        assert output.shape == x.shape
    
    def test_attention_with_mask(self):
        """Test attention with masking."""
        attn = MultiHeadAttention(dim=512, num_heads=8)
        x = torch.randn(2, 10, 512)
        mask = torch.ones(2, 8, 10, 10)
        mask[:, :, :, 5:] = 0  # Mask future positions
        
        output = attn(x, mask)
        assert output.shape == x.shape


# Property-based tests
class TestPropertyBased:
    """Property-based tests using Hypothesis."""
    
    @given(
        batch_size=st.integers(min_value=1, max_value=32),
        seq_len=st.integers(min_value=1, max_value=100),
        input_dim=st.integers(min_value=8, max_value=128)
    )
    def test_model_handles_various_inputs(self, batch_size, seq_len, input_dim):
        """Test model with randomly generated input shapes."""
        # Ensure dimensions are compatible
        hidden_dim = (input_dim // 8) * 8
        if hidden_dim < 8:
            hidden_dim = 8
        
        model = TransformerClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=2
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        output = model(x)
        
        assert output.shape == (batch_size, 2)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# Integration Tests
class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline(self, sample_dataframe):
        """Test complete pipeline from data to prediction."""
        # Split data
        splitter = DataSplitter()
        train, val, test = splitter.split(sample_dataframe, 'target')
        
        # Engineer features
        imputer = MissingValueImputer()
        train = imputer.fit_transform(train)
        
        # Create model
        model = TransformerClassifier(
            input_dim=3,
            hidden_dim=64,
            num_layers=2
        )
        
        # Make prediction
        features = torch.tensor(
            train.select(['feature_1', 'feature_2', 'feature_3'])
            .head(5)
            .to_numpy(),
            dtype=torch.float32
        )
        
        with torch.no_grad():
            output = model(features)
        
        assert output.shape == (5, 2)
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src", "--cov-report=html"])
