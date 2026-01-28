import os
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None
model_info = {}


class PredictionInput(BaseModel):
    """Input schema for single prediction - 20 features as list"""
    features: List[float] = Field(
        ...,
        example=[0.5, -1.2, 0.8, -0.3, 1.5, 0.2, -0.7, 0.9, -0.4, 1.1,
                 0.6, -0.9, 0.3, -1.0, 0.7, 0.1, -0.5, 0.4, -0.8, 1.3],
        description="List of 20 numeric features"
    )


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions - list of feature lists"""
    instances: List[List[float]] = Field(
        ...,
        example=[
            [0.5, -1.2, 0.8, -0.3, 1.5, 0.2, -0.7, 0.9, -0.4, 1.1,
             0.6, -0.9, 0.3, -1.0, 0.7, 0.1, -0.5, 0.4, -0.8, 1.3],
            [-0.5, 1.2, -0.8, 0.3, -1.5, -0.2, 0.7, -0.9, 0.4, -1.1,
             -0.6, 0.9, -0.3, 1.0, -0.7, -0.1, 0.5, -0.4, 0.8, -1.3]
        ],
        description="List of instances, each with 20 numeric features"
    )


class PredictionOutput(BaseModel):
    """Output schema for predictions"""
    churn_probability: float
    will_churn: bool
    model_version: str


class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions"""
    predictions: List[PredictionOutput]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    global model, model_info
    
    # Startup
    logger.info("Starting ChurnGuard Model Serving API")
    
    # Get MLflow configuration from environment
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
    mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME", "user")
    mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    model_name = os.getenv("MODEL_NAME", "churn-prediction-model")
    model_version = os.getenv("MODEL_VERSION", "latest")  # Use "latest" or specific version number
    
    logger.info(f"MLflow Tracking URI: {mlflow_tracking_uri}")
    logger.info(f"Model Name: {model_name}")
    logger.info(f"Model Version: {model_version}")
    
    try:
        # Set MLflow credentials if provided
        if mlflow_username and mlflow_password:
            os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Get MLflow Client
        client = mlflow.MlflowClient(tracking_uri=mlflow_tracking_uri)
        
        # Get the actual version number if "latest" is specified
        if model_version.lower() == "latest":
            logger.info(f"Finding latest version of model: {model_name}")
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise ValueError(f"No versions found for model '{model_name}'")
            # Get the highest version number
            latest_version = max([int(v.version) for v in versions])
            model_version = str(latest_version)
            logger.info(f"Latest version found: {model_version}")
        
        # Get model version details
        mv = client.get_model_version(name=model_name, version=model_version)
        logger.info(f"Found model version {model_version}, run_id: {mv.run_id}, source: {mv.source}")
        
        # Try loading model using source path from model version
        # The source contains the artifact location
        model_uri = mv.source
        logger.info(f"Loading model from source: {model_uri}")
        
        # Load the model
        model = mlflow.pyfunc.load_model(model_uri)
        
        model_info = {
            "name": model_name,
            "version": mv.version,
            "run_id": mv.run_id,
            "description": mv.description or "No description",
            "creation_timestamp": mv.creation_timestamp,
        }
        
        logger.info(f"Model loaded successfully: {model_info}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ChurnGuard Model Serving API")
    model = None


# Create FastAPI app with lifespan
app = FastAPI(
    title="ChurnGuard Model Serving API",
    description="MLflow model serving for customer churn prediction",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "service": "ChurnGuard Model Serving",
        "status": "running",
        "model": model_info.get("name", "not loaded"),
        "version": model_info.get("version", "unknown")
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for Kubernetes probes"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_info": model_info
    }


@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_info": model_info,
        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    """
    Make a single prediction
    
    Expected input format:
    {
        "features": [0.5, -1.2, 0.8, -0.3, 1.5, 0.2, -0.7, 0.9, -0.4, 1.1,
                     0.6, -0.9, 0.3, -1.0, 0.7, 0.1, -0.5, 0.4, -0.8, 1.3]
    }
    (20 numeric features)
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Validate feature count
        if len(input_data.features) != 20:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Expected 20 features, got {len(input_data.features)}"
            )
        
        # Convert input to numpy array with proper shape
        input_array = np.array([input_data.features])
        
        # Make prediction
        prediction = model.predict(input_array)
        
        # Handle different prediction formats
        if hasattr(prediction, '__iter__') and not isinstance(prediction, str):
            churn_prob = float(prediction[0])
        else:
            churn_prob = float(prediction)
        
        # Ensure probability is between 0 and 1
        churn_prob = max(0.0, min(1.0, churn_prob))
        
        return PredictionOutput(
            churn_probability=churn_prob,
            will_churn=churn_prob > 0.5,
            model_version=model_info.get("version", "unknown")
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Prediction"])
async def predict_batch(input_data: BatchPredictionInput):
    """
    Make batch predictions
    
    Expected input format:
    {
        "instances": [
            [0.5, -1.2, 0.8, ..., 1.3],  # 20 features
            [-0.5, 1.2, -0.8, ..., -1.3]  # 20 features
        ]
    }
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Validate feature counts
        for i, instance in enumerate(input_data.instances):
            if len(instance) != 20:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Instance {i}: Expected 20 features, got {len(instance)}"
                )
        
        # Convert input to numpy array
        input_array = np.array(input_data.instances)
        
        # Make predictions
        predictions = model.predict(input_array)
        
        # Format output
        results = []
        for pred in predictions:
            if hasattr(pred, '__iter__') and not isinstance(pred, str):
                churn_prob = float(pred[0]) if len(pred) > 0 else float(pred)
            else:
                churn_prob = float(pred)
            
            # Ensure probability is between 0 and 1
            churn_prob = max(0.0, min(1.0, churn_prob))
            
            results.append(
                PredictionOutput(
                    churn_probability=churn_prob,
                    will_churn=churn_prob > 0.5,
                    model_version=model_info.get("version", "unknown")
                )
            )
        
        return BatchPredictionOutput(predictions=results)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get basic metrics (can be extended for Prometheus)"""
    return {
        "model_loaded": model is not None,
        "model_version": model_info.get("version", "unknown"),
        "service": "churnguard-serving"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "serve_model:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
