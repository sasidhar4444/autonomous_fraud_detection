# Autonomous Workflow Engine

A comprehensive system that ingests transaction data → preprocesses → trains ML model → predicts fraud → triggers automated actions → runs on a scheduler → exposes FastAPI → has a Streamlit dashboard.

## Architecture

```
┌─────────────┐
│ Data Source │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Ingestion  │────▶│ Preprocessing│────▶│   Model     │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
       ┌─────────────────────────────────────────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Prediction │────▶│ Automation   │────▶│  Actions    │
└─────────────┘     └──────────────┘     │ (Email/Web) │
       │                                  └─────────────┘
       │
       ▼
┌─────────────┐
│  Scheduler  │
└─────────────┘
```

## Project Structure

```
autonomous-workflow/
├── requirements.txt          # Python dependencies
├── docker-compose.yml        # Docker orchestration
├── Dockerfile               # Docker image definition
├── .env.example             # Environment variables template
├── README.md                # This file
├── data/                    # Data directory
│   ├── sample_transactions.csv
│   └── out/
│       └── flags.csv        # Flagged transactions
├── models/                  # Model artifacts
│   ├── pipeline_v1.joblib
│   └── preprocessor.joblib
├── logs/                    # Logs directory
│   ├── run.log
│   └── metrics.csv
├── scripts/                 # Utility scripts
│   └── generate_sample_data.py
├── src/
│   ├── app/                 # Main application
│   │   ├── main.py         # FastAPI service
│   │   ├── ingestion.py    # Data ingestion
│   │   ├── preprocess.py   # Preprocessing pipeline
│   │   ├── model.py        # Model loading
│   │   ├── automation.py   # Automation engine
│   │   ├── scheduler.py    # Scheduler
│   │   └── utils.py        # Utilities
│   ├── training/           # Training pipeline
│   │   ├── train.py        # Training script
│   │   ├── features.py     # Feature engineering
│   │   └── eval.py         # Evaluation
│   └── dashboard/          # Streamlit dashboard
│       └── app.py
└── tests/                  # Test directory
```

## Features

### 1. Synthetic Data Generator
- Generates 5,000+ synthetic transaction rows
- Columns: timestamp, user_id, amount, merchant, method, country, label
- Fraud label rule: high-amount + UPI randomness

### 2. Preprocessing Pipeline
- scikit-learn ColumnTransformer
- Numeric: imputer + StandardScaler
- Categorical: imputer + OneHotEncoder
- Exports preprocessor artifact

### 3. Training Pipeline
- RandomForestClassifier with class_weight="balanced"
- Derived features (time, amount, user-based)
- Train/test split
- Logs metrics to CSV
- Saves model pipeline

### 4. FastAPI Prediction Service
- `/health` - Health check endpoint
- `/predict` - Single transaction prediction
- `/predict/batch` - Batch prediction
- Pydantic models for validation

### 5. Automation Engine
- Email alerts (SMTP)
- Webhook POST requests
- Google Sheets append (optional)
- Configurable fraud threshold

### 6. Scheduler
- APScheduler for periodic runs
- Complete workflow: ingest → preprocess → predict → automate
- Configurable interval

### 7. Streamlit Dashboard
- Flags table with filters
- Recent run logs
- Model metrics visualization

### 8. Docker Support
- Dockerfile for containerization
- docker-compose.yml for orchestration
- Separate services: API, Scheduler, Dashboard

## Setup Instructions

### 1. Prerequisites

- Python 3.11+
- Docker and Docker Compose (optional)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate Sample Data

```bash
python scripts/generate_sample_data.py
```

This creates `data/sample_transactions.csv` with 5,000 transactions.

### 4. Train the Model

```bash
python src/training/train.py
```

This will:
- Load and preprocess data
- Create derived features
- Train RandomForestClassifier
- Save model to `models/pipeline_v1.joblib`
- Save preprocessor to `models/preprocessor.joblib`
- Save metrics to `logs/metrics.csv`

### 5. Configure Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` with your settings:
- Email credentials (for alerts)
- Webhook URL (optional)
- Google Sheets credentials (optional)
- Fraud threshold

### 6. Run the Services

#### Option A: Run Locally

**FastAPI Service:**
```bash
python src/app/main.py
# Or: uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

**Scheduler:**
```bash
python src/app/scheduler.py
```

**Dashboard:**
```bash
streamlit run src/dashboard/app.py
```

#### Option B: Run with Docker

```bash
docker-compose up -d
```

This starts:
- API on http://localhost:8000
- Dashboard on http://localhost:8501
- Scheduler (runs in background)

### 7. Test the API

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Single Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-01-15T10:30:00",
    "user_id": "USER_1234",
    "amount": 1500.00,
    "merchant": "Amazon",
    "method": "UPI",
    "country": "IND"
  }'
```

**Batch Prediction:**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "timestamp": "2024-01-15T10:30:00",
        "user_id": "USER_1234",
        "amount": 1500.00,
        "merchant": "Amazon",
        "method": "UPI",
        "country": "IND"
      }
    ]
  }'
```

## Workflow Execution

The scheduler runs the complete workflow periodically:

1. **Ingest**: Read transaction data from CSV
2. **Preprocess**: Apply preprocessing pipeline
3. **Predict**: Generate fraud predictions
4. **Filter**: Identify transactions above threshold
5. **Automate**: Trigger email/webhook/Google Sheets actions
6. **Log**: Record all actions and results

## Dashboard

Access the Streamlit dashboard at http://localhost:8501

Features:
- **Flags**: View flagged transactions with filters
- **Logs**: View recent run logs
- **Metrics**: View model performance metrics

## Automation Actions

When a transaction is flagged (probability ≥ threshold):

1. **Email Alert**: Sends email via SMTP
2. **Webhook**: POSTs to configured webhook URL
3. **Google Sheets**: Appends row to Google Sheet (if enabled)

## Model Details

- **Algorithm**: RandomForestClassifier
- **Class Weight**: Balanced (handles imbalanced data)
- **Features**: 
  - Original: amount, merchant, method, country
  - Derived: time features, amount features, user statistics
- **Preprocessing**: StandardScaler for numeric, OneHotEncoder for categorical

## Logging

All actions are logged to `logs/run.log` with structured format:
- Timestamp
- Action type
- Details (counts, model version, automation results)

## Development

### Run Tests

```bash
# Add tests to tests/ directory
pytest tests/
```

### Add New Features

1. Add feature engineering in `src/training/features.py`
2. Update preprocessing in `src/app/preprocess.py`
3. Retrain model: `python src/training/train.py`

## Troubleshooting

### Model Not Found
- Ensure you've run the training script first
- Check that `models/pipeline_v1.joblib` exists

### Email Not Sending
- Verify SMTP credentials in `.env`
- Check firewall/network settings
- For Gmail, use App Password (not regular password)

### Dashboard Not Loading
- Ensure data files exist in `data/out/` and `logs/`
- Check file permissions

## License

MIT License

## Support

For issues or questions, please open an issue on the repository.






