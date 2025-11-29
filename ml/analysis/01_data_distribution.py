import os
import pandas as pd
import numpy as np
from scipy.io import arff

# Define the list of files to analyze
file_names = [
    'TimeBasedFeatures-Dataset-15s-VPN.arff',
]

# Assume files are in the current directory; modify path_to_files if not
path_to_files = 'ml/data/processed' 

def decode_byte_strings(df):
    """
    After ARFF loading, strings are often in byte format (b'value'),
    this function converts them to regular strings.
    """
    str_df = df.select_dtypes([object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        df[col] = str_df[col]
    return df

def analyze_dataset(file_path):
    print("=" * 60)
    print(f"Analyzing file: {os.path.basename(file_path)}")
    print("=" * 60)

    try:
        # Load ARFF file
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        
        # Decode byte strings
        df = decode_byte_strings(df)
        
        # 1. Basic information
        print(f"--- 1. Dataset Shape ---")
        print(f"Number of rows (samples): {df.shape[0]}")
        print(f"Number of columns (features): {df.shape[1]}")
        print("-" * 30)

        # 2. Class distribution (check dataset balance)
        # ISCX datasets usually have the label in the last column (e.g., 'class1')
        target_col = df.columns[-1] 
        print(f"--- 2. Class Distribution (Target Column: {target_col}) ---")
        class_counts = df[target_col].value_counts()
        class_percent = df[target_col].value_counts(normalize=True) * 100
        
        dist_df = pd.DataFrame({'Count': class_counts, 'Percentage (%)': class_percent})
        print(dist_df)
        
        # Check for severe class imbalance (e.g., class < 5%)
        if (class_percent < 5).any():
            print("\n[!] Warning: Severe class imbalance detected! One or more classes under 5%.")
        print("-" * 30)

        # 3. Special value detection (NaN, Inf, 0, negative values)
        print(f"--- 3. Special Value Check ---")
        
        # Select numeric columns for analysis
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Check NaN
        nan_counts = numeric_df.isna().sum().sum()
        # Check Infinity
        inf_counts = np.isinf(numeric_df).values.sum()
        
        print(f"Total NaN values: {nan_counts}")
        print(f"Total Inf values: {inf_counts}")

        # Check for zeros
        zeros = (numeric_df == 0).sum()
        zero_cols = zeros[zeros > 0]
        if not zero_cols.empty:
            print(f"\n[i] Columns containing zero values (Top 5):")
            print(zero_cols.sort_values(ascending=False).head(5))
        else:
            print("No numeric columns contain zero values.")

        # Check for negative values (network flow features normally shouldn't be negative)
        negatives = (numeric_df < 0).sum()
        neg_cols = negatives[negatives > 0]
        if not neg_cols.empty:
            print(f"\n[!] Warning: Negative values detected (may be abnormal depending on feature definition):")
            print(neg_cols.sort_values(ascending=False))
        else:
            print("\n[OK] No negative values detected.")
        print("-" * 30)

        # 4. Skewness analysis
        print(f"--- 4. Feature Skewness ---")
        # Skewness: 0 = normal, >1 or <-1 = highly skewed
        skewness = numeric_df.skew()
        
        high_skew = skewness[abs(skewness) > 1]
        
        print(f"Number of highly skewed features (|Skew| > 1): {len(high_skew)} / {len(numeric_df.columns)}")
        
        if not high_skew.empty:
            print("\nTop 5 features with highest skewness:")
            print(high_skew.abs().sort_values(ascending=False).head(5))
            print("\n(Note: Positive = right-skewed/long tail, Negative = left-skewed)")
        
        # 5. Descriptive statistics summary
        print("-" * 30)
        print(f"--- 5. Statistical Summary (Partial) ---")
        print(numeric_df.describe().loc[['mean', 'std', 'min', 'max']].iloc[:, :4]) # Show only first 4 columns to reduce output
        print("..." + f" (Total {len(numeric_df.columns)} columns)")
        
        print("\n\n")

    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
    except Exception as e:
        print(f"[ERROR] Error while processing {file_path}: {e}")

# Main program loop
if __name__ == "__main__":
    for fname in file_names:
        full_path = os.path.join(path_to_files, fname)
        analyze_dataset(full_path)
