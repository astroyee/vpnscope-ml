# fileName: data_analysis.py
import os
import sys
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Configuration ---
DATA_DIR = 'ml/data/processed'
OUTPUT_DIR = 'evaluation_results'
FILE_NAME = 'TimeBasedFeatures-Dataset-15s-VPN.arff'

# --- Utility: Logger to output to both console and file ---
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# --- Data Loading & Preprocessing ---
def decode_byte_strings(df):
    """
    ARFF files often load strings as bytes (b'value'). 
    This converts them to regular UTF-8 strings.
    """
    str_df = df.select_dtypes([object])
    if not str_df.empty:
        str_df = str_df.stack().str.decode('utf-8').unstack()
        for col in str_df:
            df[col] = str_df[col]
    return df

def load_data(file_path):
    print(f"Loading data from: {file_path} ...")
    try:
        # Handle ARFF empty string issues by replacing '' with ?
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().replace("''", "?")
        
        data, meta = arff.loadarff(io.StringIO(content))
        df = pd.DataFrame(data)
        df = decode_byte_strings(df)
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return None

# --- Analysis Module 1: Data Distribution ---
def analyze_distribution(df):
    print("\n" + "=" * 60)
    print("PART 1: Data Distribution Analysis")
    print("=" * 60)

    # 1. Basic Information
    print(f"--- 1. Dataset Shape ---")
    print(f"Number of Rows (Samples): {df.shape[0]}")
    print(f"Number of Columns (Features): {df.shape[1]}")
    
    # 2. Class Distribution
    target_col = df.columns[-1]
    print(f"\n--- 2. Class Distribution (Target: {target_col}) ---")
    if target_col in df.columns:
        class_counts = df[target_col].value_counts()
        class_percent = df[target_col].value_counts(normalize=True) * 100
        dist_df = pd.DataFrame({'Count': class_counts, 'Percentage (%)': class_percent})
        print(dist_df)
        
        if (class_percent < 5).any():
            print("\n[!] Warning: Severe class imbalance detected! One or more classes are under 5%.")
    else:
        print("Target column not found, skipping class distribution.")

    # 3. Special Value Check
    print(f"\n--- 3. Special Value Check (NaN, Inf, 0, Negative) ---")
    numeric_df = df.select_dtypes(include=[np.number])
    
    nan_counts = numeric_df.isna().sum().sum()
    inf_counts = np.isinf(numeric_df).values.sum()
    print(f"Total NaN values: {nan_counts}")
    print(f"Total Inf values: {inf_counts}")

    zeros = (numeric_df == 0).sum()
    zero_cols = zeros[zeros > 0]
    if not zero_cols.empty:
        print(f"\nColumns containing zeros (Top 5):")
        print(zero_cols.sort_values(ascending=False).head(5))
    else:
        print("No numeric columns contain zero values.")

    negatives = (numeric_df < 0).sum()
    neg_cols = negatives[negatives > 0]
    if not neg_cols.empty:
        print(f"\n[!] Warning: Negative values detected (Top 5):")
        print(neg_cols.sort_values(ascending=False).head(5))
    else:
        print("No negative values detected.")

    # 4. Skewness Analysis
    print(f"\n--- 4. Feature Skewness ---")
    skewness = numeric_df.skew()
    high_skew = skewness[abs(skewness) > 1]
    print(f"Number of highly skewed features (|Skew| > 1): {len(high_skew)} / {len(numeric_df.columns)}")
    if not high_skew.empty:
        print("Top 5 features with highest skewness:")
        print(high_skew.abs().sort_values(ascending=False).head(5))

    # 5. Statistical Summary
    print(f"\n--- 5. Statistical Summary (Partial) ---")
    print(numeric_df.describe().loc[['mean', 'std', 'min', 'max']].iloc[:, :4])
    print("..." + f" (Total {len(numeric_df.columns)} columns)")

# --- Analysis Module 2: PCA Analysis ---
def analyze_pca(df):
    print("\n" + "=" * 60)
    print("PART 2: PCA Analysis (Principal Component Analysis)")
    print("=" * 60)

    # Preprocessing: Remove label, keep numeric only
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])

    # Handle negative values and apply Log1p (PCA is sensitive to skewness)
    X[X < 0] = 0
    X_log = np.log1p(X)

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    # Fit PCA
    pca = PCA()
    pca.fit(X_scaled)
    
    # Cumulative Variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # --- Plot: Scree Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='.', linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle='-', label='95% Explained Variance')
    plt.axhline(y=0.99, color='g', linestyle=':', label='99% Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance Analysis (Scree Plot)')
    plt.grid(True)
    plt.legend()
    
    save_path = os.path.join(OUTPUT_DIR, 'pca_scree_plot.png')
    plt.savefig(save_path)
    plt.close()
    print(f"PCA Scree Plot saved to: {save_path}")

    # Output Results
    n_95 = np.argmax(cumulative_variance >= 0.95) + 1
    n_99 = np.argmax(cumulative_variance >= 0.99) + 1
    
    print(f"Original number of features: {X.shape[1]}")
    print(f"Components needed for 95% variance: {n_95}")
    print(f"Components needed for 99% variance: {n_99}")

# --- Analysis Module 3: Feature Correlation & Redundancy ---
def analyze_correlations(df):
    print("\n" + "=" * 60)
    print("PART 3: Feature Correlation & Redundancy Analysis")
    print("=" * 60)

    target_col = df.columns[-1]
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    
    # Handle negatives
    X[X < 0] = 0

    features_to_drop = set()

    # 1. Constant Columns (Variance = 0)
    print("\n[1] Checking for constant columns (Variance = 0)...")
    variances = X.var()
    constant_cols = variances[variances == 0].index.tolist()
    if constant_cols:
        print(f" -> Found {len(constant_cols)} constant columns (zero information):")
        print(f"    {constant_cols[:10]} ...") # Print only first 10
        features_to_drop.update(constant_cols)
    else:
        print(" -> No constant columns found.")

    # 2. High Correlation
    print("\n[2] Checking for highly correlated columns (Correlation > 0.95)...")
    X_clean = X.drop(columns=constant_cols)
    
    # Correlation Matrix
    corr_matrix = X_clean.corr().abs()

    # --- Plot: Heatmap ---
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', xticklabels=False, yticklabels=False)
    plt.title("Feature Correlation Matrix Heatmap")
    
    save_path = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Correlation Heatmap saved to: {save_path}")

    # Filtering
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [column for column in upper.columns if any(upper[column] > 0.95)]

    if high_corr_cols:
        print(f" -> Found {len(high_corr_cols)} highly redundant columns (recommended to remove):")
        print(f"    {high_corr_cols[:10]} ...")
        features_to_drop.update(high_corr_cols)
    else:
        print(" -> No highly correlated columns found.")

    print("-" * 50)
    print(f"Total features recommended to drop: {len(features_to_drop)}")
    print(f"Remaining useful features: {X.shape[1] - len(features_to_drop)}")

# --- Main Execution ---
def main():
    # 1. Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # 2. Setup Logging
    report_path = os.path.join(OUTPUT_DIR, 'analysis_report.txt')
    sys.stdout = Logger(report_path)

    print("Starting comprehensive data analysis report generation...")
    print(f"Report will be saved to: {report_path}")
    print(f"Plots will be saved to: {OUTPUT_DIR}/")

    # 3. Load Data
    full_path = os.path.join(DATA_DIR, FILE_NAME)
    df = load_data(full_path)

    if df is not None:
        # 4. Run Analysis Modules
        analyze_distribution(df)
        analyze_pca(df)
        analyze_correlations(df)
        
        print("\n" + "=" * 60)
        print("All analysis tasks completed successfully.")
        print("=" * 60)

if __name__ == "__main__":
    main()