import os
import io
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# 1. 必须重新定义 Pipeline 中用到的自定义类
#    (否则 joblib.load 会报错)
# ==========================================

class VPNFeaturePreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        if isinstance(X_copy, pd.DataFrame):
            num_cols = X_copy.select_dtypes(include=[np.number]).columns
            X_copy[num_cols] = X_copy[num_cols].applymap(lambda x: 0 if x < 0 else x)
            X_copy[num_cols] = np.log1p(X_copy[num_cols])
        else:
            X_copy[X_copy < 0] = 0
            X_copy = np.log1p(X_copy)
        return X_copy

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.to_drop = []
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X)
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.drop(columns=self.to_drop, errors='ignore')
        return X

# ==========================================
# 2. 数据加载函数 (复用)
# ==========================================

def map_label(label):
    label_str = str(label).lower()
    if 'non-vpn' in label_str: return 0
    if 'vpn' in label_str or 'tor' in label_str: return 1
    return 0

def load_data():
    base_dir = 'ml/data/processed/'
    file_names = [
        'TimeBasedFeatures-Dataset-15s-VPN.arff',
    ]
    data_frames = []
    print("Loading datasets for evaluation...")
    for fname in file_names:
        try:
            path = os.path.join(base_dir, fname)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().replace("''", "?")
            data, meta = arff.loadarff(io.StringIO(content))
            df_part = pd.DataFrame(data)
            for col in df_part.select_dtypes([object]).columns:
                df_part[col] = df_part[col].str.decode('utf-8')
            data_frames.append(df_part)
        except Exception: pass
    
    df = pd.concat(data_frames, ignore_index=True)
    target_col = df.columns[-1]
    df['Binary_Label'] = df[target_col].apply(map_label)
    X = df.drop(columns=[target_col, 'Binary_Label'])
    y = df['Binary_Label']
    
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    clean_feature_names = [re.sub(r'[<>\\[\\]]', '_', name) for name in X.columns]
    X.columns = clean_feature_names
    return X, y

# ==========================================
# 3. 绘图与评估逻辑
# ==========================================

def plot_confusion_matrix(y_true, y_pred, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-VPN', 'VPN'], 
                yticklabels=['Non-VPN', 'VPN'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    print(" -> Confusion Matrix saved.")

def plot_roc_curve(y_true, y_probs, save_dir):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()
    print(" -> ROC Curve saved.")

def plot_feature_importance(pipeline, feature_names, save_dir):
    # 从 Pipeline 中提取 XGBoost 模型
    # 假设 pipeline 的最后一步名为 'classifier'
    try:
        model = pipeline.named_steps['classifier']
        importances = model.feature_importances_
        
        # 确保特征名数量匹配 (feature_names 应该是经过 selector 筛选后的)
        if len(importances) != len(feature_names):
            print(f"[Warning] Feature names mismatch: Model has {len(importances)}, List has {len(feature_names)}")
            indices = np.argsort(importances)[::-1]
            names = [f"Feature {i}" for i in indices]
        else:
            indices = np.argsort(importances)[::-1]
            names = [feature_names[i] for i in indices]
            
        # 只画前 15 个重要特征
        top_n = 15
        plt.figure(figsize=(10, 8))
        plt.title(f"Top {top_n} Feature Importances")
        plt.barh(range(top_n), importances[indices][:top_n], align='center')
        plt.yticks(range(top_n), names[:top_n])
        plt.gca().invert_yaxis()
        plt.xlabel("Importance Score (Gain)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'))
        plt.close()
        print(" -> Feature Importance saved.")
        
    except Exception as e:
        print(f"[Error] Could not plot feature importance: {e}")

def main():
    # 设置保存路径
    save_dir = 'evaluation_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. 加载模型和特征名
    model_path = 'server/vpn_model_artifacts.pkl'
    
    if not os.path.exists(model_path):
        print("Error: Model file not found. Please run model.py first.")
        return

    print("Loading model pipeline...")
    pipeline = joblib.load(model_path)

    # 2. 加载并拆分数据
    X, y = load_data()
    # 必须使用与训练时相同的 random_state 才能保证这是真正的“测试集”
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nEvaluating on {len(X_test)} test samples...")

    # 3. 预测
    y_pred = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)[:, 1] # 获取 VPN 类的概率

    # 4. 文本报告
    print("\n" + "="*50)
    print("FINAL MODEL EVALUATION REPORT")
    print("="*50)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print("-" * 30)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-VPN', 'VPN']))
    print("-" * 30)

    # 5. 生成图表
    print("Generating Plots...")
    plot_confusion_matrix(y_test, y_pred, save_dir)
    plot_roc_curve(y_test, y_probs, save_dir)

    print("\n" + "="*50)
    print(f"All evaluation artifacts saved to: ./{save_dir}/")
    print("="*50)

if __name__ == "__main__":
    main()