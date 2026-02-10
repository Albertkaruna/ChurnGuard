import os
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None
model_info = {}

# ID patterns to exclude from features
ID_PATTERNS = ["id", "customer", "customername", "name", "customerid"]


def _normalize_input_keys(d: dict) -> dict:
    """Normalize keys to lowercase with underscores."""
    return {str(k).lower().strip().replace(" ", "_"): v for k, v in d.items()}


def _input_dict_to_features(d: dict) -> dict:
    """Build feature dict for model: normalize keys, drop churn/ID, handle missing values."""
    normalized = _normalize_input_keys(d)
    # Drop target and ID-like columns if present; fill None for model
    numeric_keys = {"age", "tenure", "usage_frequency", "support_calls", "payment_delay", 
                    "total_spend", "last_interaction"}
    out = {}
    for k, v in normalized.items():
        if k == "churn":
            continue
        if any(p in k for p in ID_PATTERNS):
            continue
        if v is None or (isinstance(v, float) and pd.isna(v)):
            out[k] = 0.0 if k in numeric_keys else "Unknown"
        else:
            out[k] = v
    return out


class PredictionInput(BaseModel):
    """Input schema for single prediction - customer features"""
    age: float = Field(..., example=35, description="Customer age")
    gender: str = Field(..., example="Male", description="Gender")
    tenure: float = Field(..., example=24, description="Months with company")
    usage_frequency: float = Field(..., example=15, description="Usage frequency")
    support_calls: float = Field(..., example=3, description="Number of support calls")
    payment_delay: float = Field(..., example=10, description="Payment delay in days")
    subscription_type: str = Field(..., example="Premium", description="Subscription type")
    contract_length: str = Field(..., example="Annual", description="Contract length")
    total_spend: float = Field(..., example=500.0, description="Total spend")
    last_interaction: float = Field(..., example=5, description="Days since last interaction")


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions"""
    instances: List[PredictionInput] = Field(
        ...,
        description="List of customer records for batch prediction"
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
        
        # Load the model as sklearn to get predict_proba support
        try:
            model = mlflow.sklearn.load_model(model_uri)
            logger.info("Model loaded as sklearn model (with predict_proba support)")
        except Exception as sklearn_error:
            logger.warning(f"Failed to load as sklearn model: {sklearn_error}")
            logger.info("Falling back to pyfunc model (no predict_proba)")
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

# This single line handles the /metrics endpoint and standard metrics to comptable with prometheus
Instrumentator().instrument(app).expose(app)

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
    Make a single customer churn prediction
    
    Expected input: Customer features (age, gender, tenure, contract, etc.)
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Convert input to feature dict (normalizes keys, drops churn/ID, adds engineered features)
        input_dict = _input_dict_to_features(input_data.dict())
        # Ensure all numeric columns are float for MLflow compatibility
        for k, v in input_dict.items():
            if isinstance(v, (int, float)):
                input_dict[k] = float(v)
        df = pd.DataFrame([input_dict])
        
        # Make prediction
        prediction = model.predict(df)
        
        # Try to get probability if available (unwrap MLflow model if needed)
        churn_prob = 0.5  # default
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(df)
            churn_prob = float(prediction_proba[0][1])
        elif hasattr(model, '_model_impl') and hasattr(model._model_impl, 'predict_proba'):
            # MLflow wrapped model - access underlying model
            prediction_proba = model._model_impl.predict_proba(df)
            churn_prob = float(prediction_proba[0][1])
        else:
            # No probability available, use prediction as confidence
            churn_prob = float(prediction[0])
        
        will_churn = bool(prediction[0])
        
        return PredictionOutput(
            churn_probability=churn_prob,
            will_churn=will_churn,
            model_version=model_info.get("version", "unknown")
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/from-record", response_model=PredictionOutput, tags=["Prediction"])
async def predict_from_record(record: Dict[str, Any] = Body(..., embed=False)):
    """
    Make a single prediction from a raw record (e.g. ChurnGuard CSV row).
    Keys are normalized (lowercase, underscores); churn and ID columns are dropped.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    try:
        input_dict = _input_dict_to_features(record)
        # Ensure all numeric columns are float for MLflow compatibility
        for k, v in input_dict.items():
            if isinstance(v, (int, float)):
                input_dict[k] = float(v)
        df = pd.DataFrame([input_dict])
        
        prediction = model.predict(df)
        
        # Try to get probability if available
        churn_prob = 0.5  # default
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(df)
            churn_prob = float(prediction_proba[0][1])
        elif hasattr(model, '_model_impl') and hasattr(model._model_impl, 'predict_proba'):
            prediction_proba = model._model_impl.predict_proba(df)
            churn_prob = float(prediction_proba[0][1])
        else:
            churn_prob = float(prediction[0])
        
        will_churn = bool(prediction[0])
        
        return PredictionOutput(
            churn_probability=churn_prob,
            will_churn=will_churn,
            model_version=model_info.get("version", "unknown")
        )
    except Exception as e:
        logger.error(f"Prediction from record error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Prediction"])
async def predict_batch(input_data: BatchPredictionInput):
    """
    Make batch churn predictions for multiple customers
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Convert all instances to feature dicts (ChurnGuard schema)
        instances_list = [_input_dict_to_features(instance.dict()) for instance in input_data.instances]
        # Ensure all numeric columns are float for MLflow compatibility
        for inst in instances_list:
            for k, v in inst.items():
                if isinstance(v, (int, float)):
                    inst[k] = float(v)
        df = pd.DataFrame(instances_list)
        
        # Make predictions
        predictions = model.predict(df)
        
        # Try to get probabilities if available
        predictions_proba = None
        if hasattr(model, 'predict_proba'):
            predictions_proba = model.predict_proba(df)
        elif hasattr(model, '_model_impl') and hasattr(model._model_impl, 'predict_proba'):
            predictions_proba = model._model_impl.predict_proba(df)
        
        # Format output
        results = []
        for i, pred in enumerate(predictions):
            if predictions_proba is not None:
                churn_prob = float(predictions_proba[i][1])  # Probability of class 1 (churn)
            else:
                churn_prob = float(pred)  # Use prediction as confidence
            will_churn = bool(pred)
            
            results.append(
                PredictionOutput(
                    churn_probability=churn_prob,
                    will_churn=will_churn,
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
