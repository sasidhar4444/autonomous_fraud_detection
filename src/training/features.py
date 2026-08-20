"""
Feature Engineering
Derived features for transaction data
"""
import pandas as pd
import numpy as np
from datetime import datetime

# def extract_time_features(df, timestamp_col='timestamp'):
#     """
#     Extract time-based features from timestamp
    
#     Args:
#         df: DataFrame with timestamp column
#         timestamp_col: Name of timestamp column
        
#     Returns:
#         DataFrame with additional time features
#     """
#     df = df.copy()
    
#     # Convert to datetime if string
#     if df[timestamp_col].dtype == 'object':
#         df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
#     # Extract features
#     df['hour'] = df[timestamp_col].dt.hour
#     df['day_of_week'] = df[timestamp_col].dt.dayofweek
#     df['day_of_month'] = df[timestamp_col].dt.day
#     df['month'] = df[timestamp_col].dt.month
#     df['is_weekend'] = (df[timestamp_col].dt.dayofweek >= 5).astype(int)
#     df['is_night'] = ((df[timestamp_col].dt.hour >= 22) | (df[timestamp_col].dt.hour < 6)).astype(int)
    
#     return df
# paste this entire function in place of the old extract_time_features(...)
def extract_time_features(df, timestamp_col='timestamp'):
    """
    Robust extraction of time-based features.

    Handles mixed tz-aware / tz-naive datetimes by coercing to UTC,
    then safely extracting hour/day/week features. Invalid timestamps
    are coerced to NaT and then filled (or left NaT depending on downstream policy).

    Returns dataframe with new columns added in-place.
    """
    import logging
    logger = logging.getLogger(__name__)

    if timestamp_col not in df.columns:
        logger.debug(f"extract_time_features: no column {timestamp_col} present")
        # add empty columns so downstream code doesn't crash
        df['hour'] = pd.NA
        df['dayofweek'] = pd.NA
        df['is_weekend'] = pd.NA
        return df

    # Coerce to datetime robustly, making everything timezone-aware (UTC)
    # This avoids "Tz-aware datetime.datetime cannot be converted" errors.
    # errors='coerce' will convert bad strings to NaT.
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce', utc=True)

    n_bad = int(df[timestamp_col].isna().sum())
    if n_bad:
        logger.warning(f"extract_time_features: {n_bad} invalid/missing timestamps after coercion")

    # Now we have tz-aware timestamps (UTC). Extract features from them.
    # Use .dt accessor which works with tz-aware Series.
    # Create hour (0-23), dayofweek (0=Mon), is_weekend boolean.
    try:
        df['hour'] = df[timestamp_col].dt.hour
        df['dayofweek'] = df[timestamp_col].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    except Exception as e:
        # defensive fallback: if dt access fails, fill with NA and log
        logger.exception("extract_time_features: failed to extract dt features")
        df['hour'] = pd.NA
        df['dayofweek'] = pd.NA
        df['is_weekend'] = pd.NA

    # If downstream expects tz-naive datetime64 (no tz), convert back to naive UTC
    # by dropping tz info — but keep the original column as UTC-aware if desired.
    # Here we create a separate column 'timestamp_utc' without tz to avoid surprises.
    try:
        # convert to python datetimes (naive, UTC) for compatibility
        df['timestamp_utc'] = df[timestamp_col].apply(
            lambda x: x.to_pydatetime().replace(tzinfo=None) if pd.notna(x) else pd.NaT
        )
    except Exception:
        # fallback: keep timestamp_utc as copy of timestamp_col if conversion fails
        df['timestamp_utc'] = df[timestamp_col]

    return df

def extract_amount_features(df, amount_col='amount'):
    """
    Extract amount-based features
    
    Args:
        df: DataFrame with amount column
        amount_col: Name of amount column
        
    Returns:
        DataFrame with additional amount features
    """
    df = df.copy()
    
    # Amount categories
    df['amount_log'] = np.log1p(df[amount_col])
    df['amount_sqrt'] = np.sqrt(df[amount_col])
    df['is_high_amount'] = (df[amount_col] > df[amount_col].quantile(0.9)).astype(int)
    df['is_low_amount'] = (df[amount_col] < df[amount_col].quantile(0.1)).astype(int)
    
    return df

def extract_user_features(df, user_id_col='user_id', amount_col='amount'):
    """
    Extract user-based aggregated features
    
    Args:
        df: DataFrame with user_id and amount columns
        user_id_col: Name of user_id column
        amount_col: Name of amount column
        
    Returns:
        DataFrame with additional user features
    """
    df = df.copy()
    
    # User statistics
    user_stats = df.groupby(user_id_col)[amount_col].agg([
        'mean', 'std', 'count', 'min', 'max'
    ]).reset_index()
    user_stats.columns = [user_id_col, 'user_avg_amount', 'user_std_amount', 
                          'user_txn_count', 'user_min_amount', 'user_max_amount']
    
    # Merge back
    df = df.merge(user_stats, on=user_id_col, how='left')
    
    # Fill NaN for new users
    df['user_avg_amount'] = df['user_avg_amount'].fillna(df[amount_col].mean())
    df['user_std_amount'] = df['user_std_amount'].fillna(df[amount_col].std())
    df['user_txn_count'] = df['user_txn_count'].fillna(1)
    df['user_min_amount'] = df['user_min_amount'].fillna(df[amount_col].min())
    df['user_max_amount'] = df['user_max_amount'].fillna(df[amount_col].max())
    
    # Amount deviation from user average
    df['amount_deviation'] = df[amount_col] - df['user_avg_amount']
    df['amount_deviation_pct'] = (df['amount_deviation'] / (df['user_avg_amount'] + 1e-6)) * 100
    
    return df

def create_derived_features(df):
    """
    Create all derived features
    
    Args:
        df: Original DataFrame
        
    Returns:
        DataFrame with all derived features
    """
    df = extract_time_features(df)
    df = extract_amount_features(df)
    df = extract_user_features(df)
    
    return df

if __name__ == "__main__":
    # Test feature engineering
    df = pd.read_csv("data/sample_transactions.csv")
    df_features = create_derived_features(df)
    print(f"Original columns: {len(df.columns)}")
    print(f"With features: {len(df_features.columns)}")
    print(f"New columns: {set(df_features.columns) - set(df.columns)}")

