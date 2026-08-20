"""
FastAPI Prediction Service
Main API endpoint for predictions
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import sys
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model import get_model_pipeline
from app.utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Autonomous Workflow Engine API", version="1.0.0")

# Global model pipeline (lazy loaded)
_model_pipeline = None

def get_model():
    """Get or load model pipeline"""
    global _model_pipeline
    if _model_pipeline is None:
        try:
            _model_pipeline = get_model_pipeline()
            logger.info("Model pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    return _model_pipeline

# Pydantic models for request/response
class Transaction(BaseModel):
    """Transaction input model"""
    timestamp: str
    user_id: str
    amount: float
    merchant: str
    method: str
    country: str

class PredictionRequest(BaseModel):
    """Prediction request model"""
    transactions: List[Transaction]

class PredictionResponse(BaseModel):
    """Prediction response model"""
    pred: int
    probability: float

class BatchPredictionResponse(BaseModel):
    """Batch prediction response model"""
    predictions: List[PredictionResponse]

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "autonomous-workflow-engine"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    """
    Predict fraud for a single transaction
    
    Args:
        transaction: Transaction data
        
    Returns:
        Prediction and probability
    """
    try:
        model = get_model()
        
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame([transaction.dict()])
        
        # Predict
        predictions, probabilities = model.predict(df)
        
        return PredictionResponse(
            pred=int(predictions[0]),
            probability=float(probabilities[0])
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: PredictionRequest):
    """
    Predict fraud for multiple transactions
    
    Args:
        request: Batch prediction request
        
    Returns:
        List of predictions
    """
    try:
        model = get_model()
        
        # Convert to DataFrame
        import pandas as pd
        transactions = [t.dict() for t in request.transactions]
        df = pd.DataFrame(transactions)
        
        # Predict
        predictions, probabilities = model.predict(df)
        
        # Format response
        results = [
            PredictionResponse(pred=int(p), probability=float(prob))
            for p, prob in zip(predictions, probabilities)
        ]
        
        return BatchPredictionResponse(predictions=results)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('API_PORT', '8000'))
    uvicorn.run(app, host="0.0.0.0", port=port)

