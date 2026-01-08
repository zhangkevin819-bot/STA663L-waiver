"""
Production FastAPI serving with async inference, batching, and monitoring.
"""

from typing import Optional

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
from contextlib import asynccontextmanager

from ..utils.logging import logger


# Request/Response models
class PredictionRequest(BaseModel):
    """Request schema for predictions."""
    features: list[float] = Field(..., description="Input features")
    batch: Optional[list[list[float]]] = Field(None, description="Batch of inputs")


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    prediction: int
    probabilities: list[float]
    confidence: float
    model_version: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    version: str


# Global model container
class ModelContainer:
    """Container for loaded model with lazy initialization."""
    
    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.version: str = "1.0.0"
    
    def load_model(self, model_path: str) -> None:
        """Load model from checkpoint."""
        logger.info(f"Loading model from {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Reconstruct model architecture
            # In production, this would come from config
            from ..models.architectures import TransformerClassifier
            
            self.model = TransformerClassifier(
                input_dim=checkpoint.get('input_dim', 10),
                hidden_dim=512,
                num_layers=6,
                num_heads=8
            )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    @torch.no_grad()
    def predict(self, features: np.ndarray) -> tuple[int, list[float], float]:
        """Run inference on input features."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Convert to tensor
        x = torch.from_numpy(features).float().to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Forward pass
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1)
        
        # Get prediction
        prediction = probs.argmax(dim=-1).item()
        probabilities = probs[0].cpu().numpy().tolist()
        confidence = max(probabilities)
        
        return prediction, probabilities, confidence
    
    @torch.no_grad()
    def predict_batch(
        self,
        features_batch: list[np.ndarray]
    ) -> list[tuple[int, list[float], float]]:
        """Run batched inference."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Stack into batch
        x = torch.from_numpy(np.stack(features_batch)).float().to(self.device)
        
        # Forward pass
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1)
        
        # Extract results
        predictions = probs.argmax(dim=-1).cpu().numpy()
        probabilities = probs.cpu().numpy()
        confidences = probabilities.max(axis=-1)
        
        return [
            (int(pred), probs.tolist(), float(conf))
            for pred, probs, conf in zip(predictions, probabilities, confidences)
        ]


# Initialize container
model_container = ModelContainer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for FastAPI app."""
    # Startup: Load model
    logger.info("Starting up inference service")
    
    # In production, load from environment variable or config
    # model_container.load_model("path/to/model.pt")
    
    yield
    
    # Shutdown: Cleanup
    logger.info("Shutting down inference service")


# Create FastAPI app
app = FastAPI(
    title="ML Inference API",
    description="Production-grade ML model serving with PyTorch and FastAPI",
    version="1.0.0",
    lifespan=lifespan
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__}
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_container.model is not None,
        device=model_container.device,
        version=model_container.version
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "ML Inference API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "docs": "/docs"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint."""
    try:
        features = np.array(request.features)
        
        if features.ndim != 1:
            raise HTTPException(
                status_code=400,
                detail="Features must be a 1D array"
            )
        
        prediction, probabilities, confidence = model_container.predict(features)
        
        return PredictionResponse(
            prediction=prediction,
            probabilities=probabilities,
            confidence=confidence,
            model_version=model_container.version
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
async def batch_predict(request: PredictionRequest):
    """Batch prediction endpoint."""
    if request.batch is None:
        raise HTTPException(
            status_code=400,
            detail="Batch field is required for batch predictions"
        )
    
    try:
        features_batch = [np.array(features) for features in request.batch]
        results = model_container.predict_batch(features_batch)
        
        return {
            "predictions": [
                {
                    "prediction": pred,
                    "probabilities": probs,
                    "confidence": conf
                }
                for pred, probs, conf in results
            ],
            "model_version": model_container.version
        }
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """Get model information."""
    if model_container.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Count parameters
    total_params = sum(p.numel() for p in model_container.model.parameters())
    trainable_params = sum(
        p.numel() for p in model_container.model.parameters() if p.requires_grad
    )
    
    return {
        "model_type": type(model_container.model).__name__,
        "device": model_container.device,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "version": model_container.version
    }


# Metrics endpoint (for Prometheus)
@app.get("/metrics")
async def metrics():
    """Expose metrics in Prometheus format."""
    # In production, use prometheus_client
    return {
        "requests_total": 0,
        "predictions_total": 0,
        "errors_total": 0
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
