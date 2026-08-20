"""
Scheduler Module
Runs the workflow periodically using APScheduler
"""
import os
import sys
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion import ingest_csv
from app.preprocess import Preprocessor
from app.model import ModelPipeline
from app.automation import AutomationEngine
from app.utils import setup_logging, log_action
from training.features import create_derived_features

logger = logging.getLogger(__name__)

class WorkflowScheduler:
    """Scheduler for running the workflow periodically"""
    
    def __init__(self, data_path='data/sample_transactions.csv',
                 model_path='models/pipeline_v1.joblib',
                 preprocessor_path='models/preprocessor.joblib',
                 flags_output='data/out/flags.csv'):
        """
        Initialize scheduler
        
        Args:
            data_path: Path to input data
            model_path: Path to model
            preprocessor_path: Path to preprocessor
            flags_output: Path to save flagged transactions
        """
        self.data_path = data_path
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.flags_output = flags_output
        
        self.model_pipeline = None
        self.automation = AutomationEngine()
        self.scheduler = BlockingScheduler()
        
        # Setup logging
        setup_logging()
    
    def run_workflow(self):
        """Run the complete workflow: ingest → preprocess → predict → automate"""
        try:
            log_action('workflow_start', {'data_path': self.data_path})
            
            # Step 1: Ingest
            logger.info("Step 1: Ingesting data...")
            df = ingest_csv(self.data_path)
            log_action('ingestion', {'count': len(df)})
            
            if len(df) == 0:
                logger.warning("No data to process")
                return
            
            # Step 2: Load model
            if self.model_pipeline is None:
                logger.info("Loading model pipeline...")
                self.model_pipeline = ModelPipeline(
                    self.model_path,
                    self.preprocessor_path
                )
            
            # Step 3: Predict
            logger.info("Step 2: Predicting...")
            predictions, probabilities = self.model_pipeline.predict(df)
            
            # Add predictions to dataframe
            df['prediction'] = predictions
            df['probability'] = probabilities
            
            # Step 4: Filter flagged transactions
            threshold = float(os.getenv('FRAUD_THRESHOLD', '0.7'))
            flagged_df = df[df['probability'] >= threshold].copy()
            
            log_action('prediction', {
                'total': len(df),
                'flagged': len(flagged_df),
                'threshold': threshold
            })
            
            # Step 5: Save flagged transactions
            if len(flagged_df) > 0:
                os.makedirs(os.path.dirname(self.flags_output), exist_ok=True)
                flagged_df.to_csv(self.flags_output, index=False)
                logger.info(f"Saved {len(flagged_df)} flagged transactions to {self.flags_output}")
            
            # Step 6: Trigger automation for each flagged transaction
            automation_results = []
            for idx, row in flagged_df.iterrows():
                transaction_data = row.to_dict()
                probability = row['probability']
                
                result = self.automation.trigger_automation(transaction_data, probability)
                automation_results.append(result)
            
            log_action('automation', {
                'triggered': len(automation_results),
                'results': automation_results
            })
            
            log_action('workflow_complete', {
                'total_processed': len(df),
                'flagged': len(flagged_df)
            })
            
            logger.info("Workflow completed successfully")
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}", exc_info=True)
            log_action('workflow_error', {'error': str(e)})
    
    def start(self, interval_minutes=60):
        """
        Start the scheduler
        
        Args:
            interval_minutes: Interval in minutes between runs
        """
        logger.info(f"Starting scheduler with {interval_minutes} minute interval")
        
        # Run immediately
        self.run_workflow()
        
        # Schedule periodic runs
        self.scheduler.add_job(
            self.run_workflow,
            trigger=IntervalTrigger(minutes=interval_minutes),
            id='workflow_job',
            name='Transaction Fraud Detection Workflow'
        )
        
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped")
            self.scheduler.shutdown()

if __name__ == "__main__":
    # Run scheduler
    scheduler = WorkflowScheduler()
    interval = int(os.getenv('SCHEDULER_INTERVAL_MINUTES', '60'))
    scheduler.start(interval_minutes=interval)

