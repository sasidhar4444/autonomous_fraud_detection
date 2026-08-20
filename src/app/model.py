"""
Model Loading Utilities
"""
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.preprocess import Preprocessor
from training.features import create_derived_features

class ModelPipeline:
    """Wrapper for model and preprocessor"""
    
    def __init__(self, model_path='models/pipeline_v1.joblib', 
                 preprocessor_path='models/preprocessor.joblib'):
        """
        Load model and preprocessor
        
        Args:
            model_path: Path to saved model
            preprocessor_path: Path to saved preprocessor
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None
        self.load()
    
    def load(self):
        """Load model and preprocessor from disk"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.preprocessor = Preprocessor()
        if os.path.exists(self.preprocessor_path):
            self.preprocessor.load(self.preprocessor_path)
        else:
            raise FileNotFoundError(f"Preprocessor not found at {self.preprocessor_path}")
    
    def predict(self, df):
        """
        Predict on DataFrame
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Predictions and probabilities
        """
        # Create features
        df_features = create_derived_features(df)
        
        # Preprocess
        X = self.preprocessor.transform(df_features)
        
        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities

# Global model instance
_model_pipeline = None

def get_model_pipeline():
    """Get or create global model pipeline instance"""
    global _model_pipeline
    if _model_pipeline is None:
        _model_pipeline = ModelPipeline()
    return _model_pipeline

