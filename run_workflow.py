#!/usr/bin/env python
"""
Quick start script to run the complete workflow
"""
import os
import sys

def main():
    print("=" * 60)
    print("Autonomous Workflow Engine - Quick Start")
    print("=" * 60)
    
    # Step 1: Generate data
    print("\n[1/4] Generating sample data...")
    os.system("python scripts/generate_sample_data.py")
    
    # Step 2: Train model
    print("\n[2/4] Training model...")
    os.system("python src/training/train.py")
    
    # Step 3: Run predictions
    print("\n[3/4] Running predictions...")
    os.system("python src/app/runner.py")
    
    print("\n[4/4] Workflow complete!")
    print("\nNext steps:")
    print("  - Start API: python src/app/main.py")
    print("  - Start Dashboard: streamlit run src/dashboard/app.py")
    print("  - Start Scheduler: python src/app/scheduler.py")
    print("  - Or use Docker: docker-compose up -d")

if __name__ == "__main__":
    main()

