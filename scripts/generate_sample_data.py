"""
Synthetic Transaction Data Generator
Generates 5,000+ synthetic transaction rows with fraud labels
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_synthetic_transactions(n_samples=5000, output_path="data/sample_transactions.csv"):
    """
    Generate synthetic transaction data with fraud labels.
    
    Fraud rule: high-amount + UPI randomness
    """
    np.random.seed(42)
    random.seed(42)
    
    # Base data
    merchants = ['Amazon', 'Flipkart', 'Walmart', 'Target', 'BestBuy', 
                 'Starbucks', 'McDonalds', 'Shell', 'BP', 'Walgreens',
                 'CVS', 'HomeDepot', 'Costco', 'WholeFoods', 'Nike']
    
    methods = ['Credit Card', 'Debit Card', 'UPI', 'PayPal', 'Bank Transfer', 'Cash']
    
    countries = ['USA', 'IND', 'UK', 'CAN', 'AUS', 'GER', 'FRA', 'JPN']
    
    # Generate timestamps (last 30 days)
    start_date = datetime.now() - timedelta(days=30)
    timestamps = [start_date + timedelta(
        seconds=random.randint(0, 30 * 24 * 3600)
    ) for _ in range(n_samples)]
    
    # Generate user IDs
    user_ids = [f"USER_{random.randint(1000, 9999)}" for _ in range(n_samples)]
    
    # Generate amounts (skewed distribution)
    amounts = np.random.lognormal(mean=3.5, sigma=1.2, size=n_samples)
    amounts = np.round(amounts, 2)
    
    # Generate merchants
    merchant_list = [random.choice(merchants) for _ in range(n_samples)]
    
    # Generate payment methods
    method_list = [random.choice(methods) for _ in range(n_samples)]
    
    # Generate countries
    country_list = [random.choice(countries) for _ in range(n_samples)]
    
    # Generate fraud labels based on rule: high-amount + UPI randomness
    labels = []
    for i in range(n_samples):
        is_high_amount = amounts[i] > np.percentile(amounts, 90)  # Top 10% amounts
        is_upi = method_list[i] == 'UPI'
        # Fraud if high amount AND UPI, plus some randomness
        if is_high_amount and is_upi:
            fraud_prob = 0.7  # 70% chance if high amount + UPI
        elif is_high_amount:
            fraud_prob = 0.15  # 15% chance if just high amount
        elif is_upi:
            fraud_prob = 0.05  # 5% chance if just UPI
        else:
            fraud_prob = 0.01  # 1% baseline
        
        # Add some randomness
        if random.random() < fraud_prob:
            labels.append(1)  # Fraud
        else:
            labels.append(0)  # Normal
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'user_id': user_ids,
        'amount': amounts,
        'merchant': merchant_list,
        'method': method_list,
        'country': country_list,
        'label': labels
    })
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Generated {n_samples} transactions. Saved to {output_path}")
    print(f"Fraud rate: {sum(labels) / len(labels) * 100:.2f}%")
    print(f"Fraud count: {sum(labels)}")
    
    return df

if __name__ == "__main__":
    generate_synthetic_transactions(n_samples=5000)

