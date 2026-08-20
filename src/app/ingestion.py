"""
Data Ingestion Module
Handles reading and processing transaction data
"""
import os
from datetime import datetime
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def ingest_csv(filepath):
    """
    Ingest data from CSV file

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame with transaction data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)

    # ---------- TIMESTAMP: robust parsing ----------
    if 'timestamp' in df.columns:
        # Normalize obvious empty tokens
        df['timestamp'] = df['timestamp'].replace(["", " ", "nan", "NaN", None], pd.NA)

        # Coerce invalid formats to NaT (no crash)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', infer_datetime_format=True)

        # Count and log bad timestamps
        n_bad = int(df['timestamp'].isna().sum())
        if n_bad:
            logger.warning(f"ingest_csv: {n_bad} timestamp(s) could not be parsed and were set to NaT")

        # Fill missing timestamps with current UTC time to avoid downstream failures
        # (If you prefer to keep NaT so features can mark 'missing', switch to the Option B policy)
        df['timestamp'] = df['timestamp'].fillna(pd.Timestamp.utcnow())

    # ---------- AMOUNT: safe numeric conversion ----------
    if 'amount' in df.columns:
        def _safe_float(x):
            try:
                # handle pandas/numpy types and strip commas
                if pd.isna(x):
                    return np.nan
                if isinstance(x, str):
                    x = x.strip().replace(",", "")
                return float(x)
            except Exception:
                return np.nan

        df['amount'] = df['amount'].apply(_safe_float)

    # ---------- MERCHANT / METHOD: normalize textual columns ----------
    if 'merchant' in df.columns:
        df['merchant'] = df['merchant'].astype(str).str.strip()
        # Replace obvious empty strings and 'nan' with None
        df['merchant'] = df['merchant'].replace({"": None, "nan": None, "None": None})

    if 'method' in df.columns:
        # lower-case and strip; keep None for missing
        df['method'] = df['method'].astype(str).str.lower().str.strip()
        df['method'] = df['method'].replace({"": None, "none": None, "nan": None, "nan.0": None})

    return df


def ingest_new_transactions(data_dir='data', pattern='*.csv'):
    """
    Ingest all new transaction files from directory

    Args:
        data_dir: Directory containing transaction files
        pattern: File pattern to match

    Returns:
        Combined DataFrame
    """
    import glob

    files = glob.glob(os.path.join(data_dir, pattern))
    if not files:
        return pd.DataFrame()

    dfs = []
    for file in files:
        # Skip sample data and output files explicitly
        if 'sample_transactions' in file or 'flags' in file:
            continue
        df = ingest_csv(file)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df
