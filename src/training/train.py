"""
Training Pipeline
Trains RandomForestClassifier with class_weight="balanced"
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.preprocess import Preprocessor
from training.features import create_derived_features

def train_model(data_path='data/sample_transactions.csv', 
                model_save_path='models/pipeline_v1.joblib',
                preprocessor_save_path='models/preprocessor.joblib',
                metrics_save_path='logs/metrics.csv',
                test_size=0.2,
                random_state=42):
    """
    Train RandomForestClassifier model
    
    Args:
        data_path: Path to training data
        model_save_path: Path to save trained model
        preprocessor_save_path: Path to save preprocessor
        metrics_save_path: Path to save metrics
        test_size: Test set size ratio
        random_state: Random seed
    """
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    
    print("Creating derived features...")
    df = create_derived_features(df)
    
    print("Preprocessing...")
    preprocessor = Preprocessor()
    X, y = preprocessor.fit_transform(df, target_col='label')
    
    # Save preprocessor
    preprocessor.save(preprocessor_save_path)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    # Train predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    
    # Test predictions
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_metrics = {
        'split': 'train',
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred, zero_division=0),
        'recall': recall_score(y_train, y_train_pred, zero_division=0),
        'f1': f1_score(y_train, y_train_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_train, y_train_proba) if len(np.unique(y_train)) > 1 else 0.0
    }
    
    test_metrics = {
        'split': 'test',
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1': f1_score(y_test, y_test_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) > 1 else 0.0
    }
    
    print("\nTrain Metrics:")
    for key, value in train_metrics.items():
        if key != 'split':
            print(f"  {key}: {value:.4f}")
    
    print("\nTest Metrics:")
    for key, value in test_metrics.items():
        if key != 'split':
            print(f"  {key}: {value:.4f}")
    
    # Save metrics
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    metrics_df = pd.DataFrame([train_metrics, test_metrics])
    metrics_df.to_csv(metrics_save_path, index=False)
    print(f"\nMetrics saved to {metrics_save_path}")
    
    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return model, preprocessor, test_metrics

if __name__ == "__main__":
    train_model()

