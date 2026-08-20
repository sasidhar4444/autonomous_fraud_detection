"""
Runner Script
Reads CSV, predicts all, writes flagged rows, triggers automation
"""
import os
import sys
import pandas as pd
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion import ingest_csv
from app.model import ModelPipeline
from app.automation import handle_flagged_rows
from app.utils import setup_logging, log_action
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

def run_predictions(data_path='data/sample_transactions.csv',
                   model_path='models/pipeline_v1.joblib',
                   preprocessor_path='models/preprocessor.joblib',
                   flags_output='data/out/flags.csv'):
    """
    Run predictions on data and trigger automation
    
    Args:
        data_path: Path to input CSV
        model_path: Path to model
        preprocessor_path: Path to preprocessor
        flags_output: Path to save flagged transactions
    """
    try:
        log_action('runner_start', {'data_path': data_path})
        
        # Step 1: Read CSV
        logger.info(f"Reading data from {data_path}")
        df = ingest_csv(data_path)
        logger.info(f"Loaded {len(df)} transactions")
        
        if len(df) == 0:
            logger.warning("No data to process")
            return
        
        # Step 2: Load model
        logger.info("Loading model pipeline...")
        model_pipeline = ModelPipeline(model_path, preprocessor_path)
        
        # Step 3: Predict all
        logger.info("Running predictions...")
        predictions, probabilities = model_pipeline.predict(df)
        
        # Add predictions to dataframe
        df['prediction'] = predictions
        df['probability'] = probabilities
        
        # Step 4: Filter flagged transactions
        threshold = float(os.getenv('FRAUD_THRESHOLD', '0.7'))
        flagged_df = df[df['probability'] >= threshold].copy()
        
        logger.info(f"Flagged {len(flagged_df)} out of {len(df)} transactions (threshold: {threshold})")
        
        # Step 5: Write flagged rows to CSV
        if len(flagged_df) > 0:
            os.makedirs(os.path.dirname(flags_output), exist_ok=True)
            flagged_df.to_csv(flags_output, index=False)
            logger.info(f"Saved flagged transactions to {flags_output}")
        else:
            # Create empty file with headers
            os.makedirs(os.path.dirname(flags_output), exist_ok=True)
            df.head(0).to_csv(flags_output, index=False)
            logger.info("No flagged transactions to save")
        
        # Step 6: Trigger automation engine (batched)
        logger.info("Triggering automation engine...")
        
        # Build run metadata
        run_meta = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'model_path': model_path,
            'threshold': threshold,
            'total_flagged': len(flagged_df)
        }
        
        # Convert flagged rows to list-of-dicts
        flagged_rows = flagged_df.to_dict(orient='records')
        
        # Call batched automation handler
        automation_results = handle_flagged_rows(flagged_rows, run_meta)
        
        log_action('automation_complete', {
            'total_flagged': len(flagged_df),
            'automation_results': automation_results
        })
        
        log_action('runner_complete', {
            'total_processed': len(df),
            'flagged': len(flagged_df),
            'output_file': flags_output
        })
        
        logger.info("Runner completed successfully")
        
        return {
            'total': len(df),
            'flagged': len(flagged_df),
            'output_file': flags_output
        }
        
    except Exception as e:
        logger.error(f"Runner failed: {e}", exc_info=True)
        log_action('runner_error', {'error': str(e)})
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run fraud detection predictions')
    parser.add_argument('--data', default='data/sample_transactions.csv',
                       help='Path to input CSV file')
    parser.add_argument('--model', default='models/pipeline_v1.joblib',
                       help='Path to model file')
    parser.add_argument('--preprocessor', default='models/preprocessor.joblib',
                       help='Path to preprocessor file')
    parser.add_argument('--output', default='data/out/flags.csv',
                       help='Path to output flagged transactions')
    
    args = parser.parse_args()
    
    run_predictions(
        data_path=args.data,
        model_path=args.model,
        preprocessor_path=args.preprocessor,
        flags_output=args.output
    )
