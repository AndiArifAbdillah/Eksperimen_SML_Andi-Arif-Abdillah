import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse
import os

def generate_dataset(n_samples=10000, seed=42):
    """Generate synthetic credit scoring dataset"""
    np.random.seed(seed)
    
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.exponential(50000, n_samples),
        'credit_score': np.random.normal(650, 100, n_samples),
        'employment_length': np.random.exponential(5, n_samples),
        'debt_to_income': np.random.beta(2, 5, n_samples) * 100,
        'loan_amount': np.random.uniform(1000, 100000, n_samples),
        'loan_term': np.random.choice([12, 24, 36, 48, 60], n_samples),
        'previous_defaults': np.random.poisson(0.3, n_samples),
        'credit_inquiries': np.random.poisson(2, n_samples),
        'credit_history_months': np.random.uniform(12, 240, n_samples),
        'savings_balance': np.random.exponential(10000, n_samples),
        'checking_balance': np.random.exponential(5000, n_samples)
    }
    
    risk_score = (
        data['age'] * 0.1 +
        data['income'] * 0.00001 +
        data['credit_score'] * 0.002 +
        data['employment_length'] * 0.05 -
        data['debt_to_income'] * 0.3 -
        data['previous_defaults'] * 50 -
        data['credit_inquiries'] * 10 +
        data['credit_history_months'] * 0.1 +
        np.random.normal(0, 10, n_samples)
    )
    
    data['default'] = (risk_score < 500).astype(int)
    
    return pd.DataFrame(data)

def preprocess_automated(input_path=None, output_path='preprocessed_data.csv'):
    """
    Automated preprocessing function
    
    Parameters:
    input_path: Path to raw data (if None, generates synthetic data)
    output_path: Path to save preprocessed data
    """
    
    if input_path and os.path.exists(input_path):
        df = pd.read_csv(input_path)
    else:
        df = generate_dataset(10000)
        print("Generated synthetic dataset")
    
    # Handle outliers
    for col in ['income', 'credit_score', 'debt_to_income']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    
    # Feature scaling
    scaler = StandardScaler()
    numerical_cols = [col for col in df.columns if col != 'default']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Save preprocessed data
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
    
    # Print dataset info
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['default'].value_counts()}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automated data preprocessing')
    parser.add_argument('--input', type=str, help='Input CSV file path')
    parser.add_argument('--output', type=str, default='credit_scoring_preprocessed.csv', 
                       help='Output CSV file path')
    
    args = parser.parse_args()
    preprocess_automated(args.input, args.output)