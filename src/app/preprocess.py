"""
Preprocessing Pipeline
Uses scikit-learn ColumnTransformer for numeric and categorical features
"""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import os

class Preprocessor:
    """Preprocessing pipeline with ColumnTransformer"""
    
    def __init__(self):
        self.preprocessor = None
        self.feature_names = None
        
    def fit_transform(self, df, target_col='label'):
        """
        Fit and transform the data
        
        Args:
            df: DataFrame with transaction data
            target_col: Name of target column
            
        Returns:
            Transformed feature array and target array
        """
        # Separate features and target
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col].values
        else:
            X = df.copy()
            y = None
        
        # Define numeric and categorical columns
        numeric_cols = ['amount']
        categorical_cols = ['merchant', 'method', 'country']
        
        # For numeric: imputer + scaler (using Pipeline)
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # For categorical: imputer + one-hot encoder (using Pipeline)
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine numeric and categorical using ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, numeric_cols),
                ('cat', categorical_pipeline, categorical_cols)
            ],
            remainder='drop'
        )
        
        # Fit and transform
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Get feature names
        try:
            numeric_features = numeric_cols
            cat_transformer = self.preprocessor.named_transformers_['cat']
            cat_features = cat_transformer.named_steps['onehot'].get_feature_names_out(categorical_cols)
            self.feature_names = list(numeric_features) + list(cat_features)
        except Exception as e:
            self.feature_names = [f'feature_{i}' for i in range(X_transformed.shape[1])]
        
        if y is not None:
            return X_transformed, y
        return X_transformed
    
    def transform(self, df):
        """Transform new data using fitted preprocessor"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor must be fitted first")
        
        X_transformed = self.preprocessor.transform(df)
        return X_transformed
    
    def save(self, filepath):
        """Save preprocessor to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.preprocessor, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load(self, filepath):
        """Load preprocessor from disk"""
        self.preprocessor = joblib.load(filepath)
        print(f"Preprocessor loaded from {filepath}")

def create_preprocessor(df, target_col='label', save_path='models/preprocessor.joblib'):
    """
    Create and save preprocessor
    
    Args:
        df: DataFrame with transaction data
        target_col: Name of target column
        save_path: Path to save preprocessor
        
    Returns:
        Preprocessor instance and transformed data
    """
    preprocessor = Preprocessor()
    X_transformed, y = preprocessor.fit_transform(df, target_col)
    preprocessor.save(save_path)
    return preprocessor, X_transformed, y

if __name__ == "__main__":
    # Test preprocessor
    df = pd.read_csv("data/sample_transactions.csv")
    preprocessor, X, y = create_preprocessor(df)
    print(f"Transformed shape: {X.shape}")
    print(f"Target shape: {y.shape}")

