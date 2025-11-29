import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Reuse previous loading logic ---
def load_data_for_analysis():
    # Make sure the path is correct
    base_dir = 'ml/data/processed/' 
    file_names = [
        'TimeBasedFeatures-Dataset-15s-VPN.arff',
    ]
    data_frames = []
    print("Loading data...")
    for fname in file_names:
        try:
            path = os.path.join(base_dir, fname)
            # Handle empty string issues
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().replace("''", "?")
            from io import StringIO
            data, meta = arff.loadarff(StringIO(content))
            df = pd.DataFrame(data)
            data_frames.append(df)
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            
    df = pd.concat(data_frames, ignore_index=True)
    
    # Remove non-numeric columns and label column
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])  # Keep numeric columns only
    
    # Preprocessing: convert negative values to 0 + Log1p (very important; PCA is sensitive to skewness)
    X[X < 0] = 0
    X = np.log1p(X)
    
    return X

def analyze_pca():
    X = load_data_for_analysis()
    
    # 1. Standardization is required before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Fit PCA
    pca = PCA()
    pca.fit(X_scaled)
    
    # 3. Compute cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # 4. Plot result
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle='-', label='95% Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance Analysis (Scree Plot)')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # 5. Output numbers
    n_95 = np.argmax(cumulative_variance >= 0.95) + 1
    n_99 = np.argmax(cumulative_variance >= 0.99) + 1
    
    print("-" * 40)
    print(f"Original number of features: {X.shape[1]}")
    print(f"Number of components needed for 95% variance: {n_95}")
    print(f"Number of components needed for 99% variance: {n_99}")
    print("-" * 40)

if __name__ == "__main__":
    analyze_pca()
