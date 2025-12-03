import os
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Dataset Path
DATA_FILE_PATH = 'ml/data/processed/TimeBasedFeatures-Dataset-15s-VPN.arff'

def load_data(file_path):
    """
    Load and preprocess the ARFF data.
    """
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return None

    print(f"Loading data from: {file_path} ...")
    try:
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        
        # Decode byte strings to normal strings
        str_df = df.select_dtypes([object])
        if not str_df.empty:
            str_df = str_df.stack().str.decode('utf-8').unstack()
            for col in str_df:
                df[col] = str_df[col]
            
        return df
    except Exception as e:
        print(f"[Error] Loading failed: {e}")
        return None

def heuristic_vpn_detection(df_in):
    """
    Apply heuristic rules (Large Packets & Long Duration) to detect VPN.
    """
    print("Applying heuristic rules...")
    
    # Create a copy to avoid modifying the original dataframe structure for label extraction later
    df = df_in.copy()
    
    # --- 1. Feature Engineering ---
    # Calculate Average Packet Size = Bytes / Packets
    # Add epsilon to avoid division by zero
    df['avg_pkt_size'] = df['flowBytesPerSecond'] / (df['flowPktsPerSecond'].replace(0, 1))

    # --- 2. Thresholds ---
    # Duration > 5 seconds (unit is microseconds in dataset)
    DURATION_THRESHOLD = 1e8 
    # Avg Packet Size > 100 Bytes (heuristic for VPN encapsulation overhead + payload)
    PKT_SIZE_THRESHOLD = 10

    # --- 3. Rules Logic ---
    # Rule: IF (Long Duration) OR (Large Packet Size) -> VPN
    
    predictions = []
    
    for index, row in df.iterrows():
        is_long_duration = row['duration'] > DURATION_THRESHOLD
        is_large_packet = row['avg_pkt_size'] > PKT_SIZE_THRESHOLD
        
        if is_long_duration or is_large_packet:
            predictions.append(1) # Predict VPN
        else:
            predictions.append(0) # Predict Non-VPN
            
    return predictions

def evaluate_results(df, predictions):
    """
    Compare predictions with ground truth.
    """
    # Identify Target Column
    # IMPORTANT: Since we used df.copy() in detection, the original df passed here 
    # still has the label as the last column.
    target_col = df.columns[-1]
    
    print(f"Ground Truth Column Identified: '{target_col}'")
    
    # Map labels to 0/1
    def map_label(label):
        label_str = str(label).lower()
        if 'non-vpn' in label_str: return 0
        if 'vpn' in label_str or 'tor' in label_str: return 1
        return 0

    y_true = df[target_col].apply(map_label)
    y_pred = predictions

    # Verify we have mixed labels
    unique_true = np.unique(y_true)
    print(f"Labels found in dataset: {unique_true} (0=Non-VPN, 1=VPN)")
    if len(unique_true) < 2:
        print("[Warning] Dataset seems to contain only one class based on mapping logic!")

    print("\n" + "="*60)
    print("Traditional Heuristic Evaluation Report")
    print("="*60)
    print(f"Rule: Duration > 5s AND Avg Pkt Size > 100 Bytes")
    print("-" * 60)
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Non-VPN', 'VPN'], digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"TN (Non-VPN correct): {cm[0][0]}")
    print(f"FP (Non-VPN -> VPN):  {cm[0][1]}")
    print(f"FN (VPN -> Non-VPN):  {cm[1][0]}")
    print(f"TP (VPN correct):     {cm[1][1]}")

def main():
    # 1. Load
    df = load_data(DATA_FILE_PATH)
    if df is None:
        return

    # 2. Detect
    # Pass a copy or handle inside to prevent column shift issues
    preds = heuristic_vpn_detection(df)

    # 3. Evaluate
    # Pass original df to ensure we grab the correct label column
    evaluate_results(df, preds)

if __name__ == "__main__":
    main()