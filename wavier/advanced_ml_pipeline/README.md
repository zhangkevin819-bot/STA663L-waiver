# Advanced ML Pipeline - Production-Grade Data Science Framework

A sophisticated, modular machine learning pipeline implementing industry best practices from data ingestion to model deployment.

## Architecture Overview

```
advanced_ml_pipeline/
├── src/
│   ├── data/              # Data engineering & ETL
│   ├── features/          # Feature engineering pipelines
│   ├── models/            # Model architectures & training
│   ├── inference/         # Serving & batch prediction
│   └── utils/             # Shared utilities
├── configs/               # Hydra configuration files
├── notebooks/             # Exploratory analysis
├── tests/                 # Comprehensive test suite
├── docker/                # Containerization
└── deployment/            # FastAPI & cloud deployment

```

## Key Features

### 🔧 Engineering Excellence
- **Type Safety**: Full type hints with mypy validation
- **Configuration**: Hydra for hierarchical config management
- **Logging**: Structured logging with contextual information
- **Testing**: Pytest with property-based testing

### 🚀 ML Capabilities
- **Data Processing**: Polars for high-performance dataframes
- **Feature Engineering**: Custom transformers with sklearn pipelines
- **Deep Learning**: PyTorch with modern architectures (Transformers, CNNs)
- **Probabilistic ML**: PyMC3 for Bayesian inference
- **Optimization**: Advanced optimizers (AdamW, Lion) with lr scheduling

### 📊 MLOps Integration
- **Experiment Tracking**: MLflow integration
- **Model Registry**: Versioned model artifacts
- **API Serving**: FastAPI with async support
- **Containerization**: Multi-stage Docker builds
- **Monitoring**: Prometheus metrics & health checks

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python src/main.py

# Start API server
uvicorn src.inference.api:app --reload

# Run tests
pytest tests/ -v --cov=src
```

## Technology Stack

- **Core**: Python 3.11+, NumPy, Polars
- **ML**: PyTorch, Scikit-learn, XGBoost, PyMC3
- **Serving**: FastAPI, Uvicorn, Docker
- **Ops**: Hydra-core, MLflow, Pydantic

## Project Structure Philosophy

This project demonstrates:
1. Separation of concerns through modular architecture
2. Dependency injection via configuration
3. Testable, composable components
4. Production-ready error handling
5. Scalable data processing patterns
