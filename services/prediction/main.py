"""
ML Prediction Microservice
FastAPI-based real-time inference service with PyTorch models
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
import numpy as np
import redis
import json
import time
import logging
from datetime import datetime
import uuid
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response

from models.model_loader import ModelLoader
from models.predictor import Predictor
from core.config import settings
from core.cache import CacheManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Prediction Service",
    description="High-performance microservice for real-time ML predictions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

INFERENCE_LATENCY = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration',
    ['model_version']
)

CACHE_HITS = Counter('cache_hits_total', 'Total cache hits')
CACHE_MISSES = Counter('cache_misses_total', 'Total cache misses')

ACTIVE_REQUESTS = Gauge('active_requests', 'Number of active requests')

# Global instances
model_loader = None
predictor = None
cache_manager = None


# Request/Response Models
class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="Input features for prediction")
    model_version: Optional[str] = Field("v1.0", description="Model version to use")
    batch_id: Optional[str] = Field(None, description="Batch identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [1.2, 3.4, 5.6, 7.8],
                "model_version": "v1.0"
            }
        }


class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_version: str
    inference_time_ms: float
    request_id: str
    timestamp: str
    cached: bool = False


class BatchPredictionRequest(BaseModel):
    batch: List[List[float]] = Field(..., description="Batch of feature arrays")
    model_version: Optional[str] = Field("v1.0", description="Model version")


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    uptime_seconds: float
    model_loaded: bool
    cache_connected: bool


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global model_loader, predictor, cache_manager
    
    logger.info("Starting ML Prediction Service...")
    
    try:
        # Initialize cache manager
        cache_manager = CacheManager(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB
        )
        logger.info("✓ Cache manager initialized")
        
        # Initialize model loader
        model_loader = ModelLoader(model_path=settings.MODEL_PATH)
        model = model_loader.load_model()
        logger.info(f"✓ Model loaded from {settings.MODEL_PATH}")
        
        # Initialize predictor
        predictor = Predictor(
            model=model,
            device=settings.DEVICE,
            batch_size=settings.BATCH_SIZE
        )
        logger.info(f"✓ Predictor initialized (device: {settings.DEVICE})")
        
        logger.info("ML Prediction Service started successfully!")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}", exc_info=True)
        raise


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down ML Prediction Service...")
    if cache_manager:
        cache_manager.close()
    logger.info("Service shutdown complete")


# Health check endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check"""
    uptime = time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    
    return HealthResponse(
        status="healthy",
        service="ml-prediction",
        version="1.0.0",
        uptime_seconds=uptime,
        model_loaded=predictor is not None,
        cache_connected=cache_manager is not None and cache_manager.is_connected()
    )


@app.get("/ready")
async def readiness_check():
    """Readiness probe for Kubernetes"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}


@app.get("/alive")
async def liveness_check():
    """Liveness probe for Kubernetes"""
    return {"status": "alive"}


# Prediction endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a single prediction
    
    - **features**: Input feature array
    - **model_version**: Model version to use (default: v1.0)
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        ACTIVE_REQUESTS.inc()
        
        # Check cache
        cache_key = cache_manager.generate_key(request.features, request.model_version)
        cached_result = cache_manager.get(cache_key)
        
        if cached_result:
            CACHE_HITS.inc()
            logger.info(f"Cache hit for request {request_id}")
            cached_result['cached'] = True
            cached_result['request_id'] = request_id
            return PredictionResponse(**cached_result)
        
        CACHE_MISSES.inc()
        
        # Convert features to numpy array
        features_array = np.array(request.features, dtype=np.float32)
        
        # Make prediction
        inference_start = time.time()
        prediction, confidence = predictor.predict(features_array)
        inference_time = (time.time() - inference_start) * 1000
        
        # Record metrics
        INFERENCE_LATENCY.labels(model_version=request.model_version).observe(
            inference_time / 1000
        )
        
        # Prepare response
        response = {
            "prediction": float(prediction),
            "confidence": float(confidence),
            "model_version": request.model_version,
            "inference_time_ms": round(inference_time, 2),
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "cached": False
        }
        
        # Cache result
        cache_manager.set(cache_key, response, ttl=settings.CACHE_TTL)
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"Prediction completed in {total_time:.2f}ms (request_id: {request_id})")
        
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status=200).inc()
        REQUEST_LATENCY.labels(method='POST', endpoint='/predict').observe(total_time / 1000)
        
        return PredictionResponse(**response)
        
    except Exception as e:
        logger.error(f"Prediction error (request_id: {request_id}): {str(e)}", exc_info=True)
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status=500).inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """
    Make batch predictions for improved throughput
    
    - **batch**: List of feature arrays
    - **model_version**: Model version to use
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        if len(request.batch) == 0:
            raise HTTPException(status_code=400, detail="Batch cannot be empty")
        
        if len(request.batch) > settings.MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size exceeds maximum of {settings.MAX_BATCH_SIZE}"
            )
        
        # Convert to numpy array
        batch_array = np.array(request.batch, dtype=np.float32)
        
        # Make batch predictions
        predictions, confidences = predictor.predict_batch(batch_array)
        
        # Prepare results
        results = [
            {
                "prediction": float(pred),
                "confidence": float(conf),
                "index": idx
            }
            for idx, (pred, conf) in enumerate(zip(predictions, confidences))
        ]
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            "results": results,
            "batch_size": len(request.batch),
            "total_time_ms": round(total_time, 2),
            "avg_time_per_sample_ms": round(total_time / len(request.batch), 2),
            "model_version": request.model_version,
            "request_id": request_id
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_version": "v1.0",
        "model_path": settings.MODEL_PATH,
        "device": str(settings.DEVICE),
        "input_shape": predictor.input_shape,
        "output_shape": predictor.output_shape,
        "parameters": predictor.count_parameters()
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    app.state.start_time = time.time()
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        workers=settings.WORKERS,
        log_level="info"
    )
