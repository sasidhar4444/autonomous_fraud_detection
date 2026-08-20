"""
Model Evaluation Utilities
"""
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.preprocess import Preprocessor
from training.features import create_derived_features

def evaluate_model(model_path='models/pipeline_v1.joblib',
                   preprocessor_path='models/preprocessor.joblib',
                   data_path='data/sample_transactions.csv'):
    """
    Evaluate model on data
    
    Args:
        model_path: Path to saved model
        preprocessor_path: Path to saved preprocessor
        data_path: Path to evaluation data
    """
    print("Loading model and preprocessor...")
    model = joblib.load(model_path)
    preprocessor = Preprocessor()
    preprocessor.load(preprocessor_path)
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    df = create_derived_features(df)
    
    print("Preprocessing...")
    X, y = preprocessor.transform(df), df['label'].values
    
    print("Predicting...")
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    
    return y_pred, y_proba

if __name__ == "__main__":
    evaluate_model()

