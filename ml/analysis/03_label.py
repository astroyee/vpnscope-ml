import os
import io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import arff

# --- 1. Load data (reuse previous logic) ---
def load_data():
    base_dir = 'ml/data/processed/'
    file_names = [
        'TimeBasedFeatures-Dataset-15s-VPN.arff',
    ]
    data_frames = []
    print("Loading data...")
    for fname in file_names:
        path = os.path.join(base_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().replace("''", "?")
            data, meta = arff.loadarff(io.StringIO(content))
            df = pd.DataFrame(data)
            data_frames.append(df)
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            
    df = pd.concat(data_frames, ignore_index=True)
    
    # Remove label column
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Preprocessing: convert negative values to 0 (avoid corrupting correlation computation)
    X[X < 0] = 0
    
    return X

def find_useless_features():
    df = load_data()
    print(f"\nOriginal number of features: {df.shape[1]}")
    
    features_to_drop = set()

    # --- Check 1: Zero-variance columns (constant features) ---
    print("\n[1] Checking constant columns (Variance = 0)...")
    variances = df.var()
    constant_cols = variances[variances == 0].index.tolist()
    if constant_cols:
        print(f" -> Found {len(constant_cols)} constant columns (no information):")
        print(f"    {constant_cols}")
        features_to_drop.update(constant_cols)
    else:
        print(" -> No constant columns found.")

    # --- Check 2: Highly correlated columns (Correlation > 0.95) ---
    print("\n[2] Checking highly correlated columns (Correlation > 0.95)...")
    # Remove constant columns before correlation computation
    df_clean = df.drop(columns=constant_cols)
    corr_matrix = df_clean.corr().abs()
    
    # Draw heatmap (optional, for visualization)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm')
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()

    # Scan upper triangle of matrix for correlations > 0.95
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    if high_corr_cols:
        print(f" -> Found {len(high_corr_cols)} highly redundant columns (to be removed):")
        for col in high_corr_cols:
            # Identify the features correlated with it
            related_cols = upper.index[upper[col] > 0.95].tolist()
            print(f"    Removing '{col}' (highly correlated with {related_cols})")
        features_to_drop.update(high_corr_cols)
    else:
        print(" -> No highly correlated columns found.")

    # --- Summary ---
    print("-" * 50)
    print(f"Total recommended features to drop: {len(features_to_drop)}")
    print(f"Recommended drop list: {list(features_to_drop)}")
    print(f"Remaining useful features: {df.shape[1] - len(features_to_drop)}")
    print("-" * 50)

if __name__ == "__main__":
    find_useless_features()
