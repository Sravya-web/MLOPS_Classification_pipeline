"""FastAPI backend for ML classification predictions with monitoring."""

import os
import joblib
import numpy as np
import logging
from typing import Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest
from dotenv import load_dotenv
from data_manager import DataManager

load_dotenv()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Classification API",
    description="Classification prediction API with MLOps best practices",
    version="1.0.0"
)

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
prediction_counter = Counter(
    'predictions_total',
    'Total number of predictions',
    ['class']
)
prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds'
)
request_counter = Counter(
    'requests_total',
    'Total requests',
    ['endpoint', 'method', 'status']
)
model_accuracy = Histogram(
    'model_accuracy',
    'Model accuracy metrics'
)

# Load model
try:
    MODEL_PATH = os.getenv("MODEL_PATH", "model_artifacts/model.joblib")
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# Initialize data manager (optional - for saving predictions)
try:
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://user:password@localhost/mlops_db"
    )
    data_manager = DataManager(db_url)
except Exception as e:
    logger.warning(f"Database connection failed: {e}")
    data_manager = None


# ==================== Pydantic Models ====================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    model_loaded: bool


class PredictionInput(BaseModel):
    """Input for prediction endpoint."""
    feature1: float = Field(..., description="Feature 1 value")
    feature2: float = Field(..., description="Feature 2 value")
    feature3: float = Field(..., description="Feature 3 value")
    feature4: float = Field(..., description="Feature 4 value")
    feature5: float = Field(..., description="Feature 5 value")

    class Config:
        json_schema_extra = {
            "example": {
                "feature1": 1.2,
                "feature2": 3.4,
                "feature3": 5.6,
                "feature4": 2.1,
                "feature5": 1.5
            }
        }


class PredictionOutput(BaseModel):
    """Output for prediction endpoint."""
    prediction: int = Field(..., description="Predicted class (0 or 1)")
    confidence: float = Field(..., description="Confidence score")
    probability_class_0: float = Field(..., description="Probability for class 0")
    probability_class_1: float = Field(..., description="Probability for class 1")
    timestamp: str = Field(..., description="Prediction timestamp")


class MetricsResponse(BaseModel):
    """Metrics response."""
    total_predictions: int
    metrics_data: str


# ==================== Health Check ====================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint.
    
    Returns:
        Health status and model availability
    """
    request_counter.labels(endpoint="/health", method="GET", status="200").inc()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        model_loaded=model is not None
    )


# ==================== Predictions ====================

@app.post("/predict", response_model=PredictionOutput, tags=["Predictions"])
def predict(input_data: PredictionInput, background_tasks: BackgroundTasks):
    """Make a prediction.
    
    Args:
        input_data: Features for prediction
        background_tasks: Background task runner
        
    Returns:
        Prediction result with confidence
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if model is None:
        request_counter.labels(endpoint="/predict", method="POST", status="500").inc()
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        with prediction_latency.time():
            # Prepare features
            features = np.array([[
                input_data.feature1,
                input_data.feature2,
                input_data.feature3,
                input_data.feature4,
                input_data.feature5
            ]])

            # Make prediction
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]

            # Update metrics
            prediction_counter.labels(class_=int(prediction)).inc()
            request_counter.labels(endpoint="/predict", method="POST", status="200").inc()

            # Prepare response
            response = PredictionOutput(
                prediction=int(prediction),
                confidence=float(max(probabilities)),
                probability_class_0=float(probabilities[0]),
                probability_class_1=float(probabilities[1]),
                timestamp=datetime.utcnow().isoformat()
            )

            # Save to database in background (if available)
            if data_manager:
                background_tasks.add_task(
                    data_manager.save_prediction,
                    [
                        input_data.feature1,
                        input_data.feature2,
                        input_data.feature3,
                        input_data.feature4,
                        input_data.feature5
                    ],
                    int(prediction),
                    float(max(probabilities))
                )

            logger.info(f"Prediction: {prediction}, Confidence: {max(probabilities):.4f}")
            return response

    except Exception as e:
        request_counter.labels(endpoint="/predict", method="POST", status="400").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ==================== Batch Predictions ====================

@app.post("/predict-batch", tags=["Predictions"])
def predict_batch(inputs: list[PredictionInput]):
    """Make batch predictions.
    
    Args:
        inputs: List of prediction inputs
        
    Returns:
        List of predictions
    """
    if model is None:
        request_counter.labels(endpoint="/predict-batch", method="POST", status="500").inc()
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Prepare features
        features = np.array([
            [
                inp.feature1, inp.feature2, inp.feature3,
                inp.feature4, inp.feature5
            ]
            for inp in inputs
        ])

        # Make predictions
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)

        # Format responses
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            prediction_counter.labels(class_=int(pred)).inc()
            results.append({
                "index": i,
                "prediction": int(pred),
                "confidence": float(max(probs)),
                "probability_class_0": float(probs[0]),
                "probability_class_1": float(probs[1])
            })

        request_counter.labels(endpoint="/predict-batch", method="POST", status="200").inc()
        return results

    except Exception as e:
        request_counter.labels(endpoint="/predict-batch", method="POST", status="400").inc()
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ==================== Metrics ====================

@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Return Prometheus metrics.
    
    Returns:
        Prometheus formatted metrics
    """
    request_counter.labels(endpoint="/metrics", method="GET", status="200").inc()
    return generate_latest()


# ==================== Model Info ====================

@app.get("/info", tags=["Information"])
def get_model_info():
    """Get model information.
    
    Returns:
        Model details and configuration
    """
    return {
        "model_name": "Random Forest Classifier",
        "version": "1.0.0",
        "features": ["feature1", "feature2", "feature3", "feature4", "feature5"],
        "classes": [0, 1],
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }


# ==================== Root ====================

@app.get("/", tags=["Root"])
def root():
    """Root endpoint."""
    return {
        "message": "ML Classification API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
