import os
import re
import io
import time
import warnings
import joblib
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy.io import arff

# Scikit-learn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_curve, 
    roc_curve, auc, roc_auc_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Boosting Models
import xgboost as xgb
import lightgbm as lgb

# Configuration
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# 1. Feature Engineering Classes
# ==========================================

class VPNFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """
    Handles initial cleaning:
    1. Clips negative values to 0.
    2. Applies Log1p transformation to compress magnitude.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if isinstance(X_copy, pd.DataFrame):
            num_cols = X_copy.select_dtypes(include=[np.number]).columns
            # Fix: Use .map instead of .applymap (deprecated)
            X_copy[num_cols] = X_copy[num_cols].map(lambda x: 0 if x < 0 else x)
            X_copy[num_cols] = np.log1p(X_copy[num_cols])
        else:
            X_copy[X_copy < 0] = 0
            X_copy = np.log1p(X_copy)
        return X_copy

class CorrelationSelector(BaseEstimator, TransformerMixin):
    """
    Removes highly correlated features to reduce multicollinearity.
    """
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
# 2. Data Loading & Helper Functions
# ==========================================

def map_label(label):
    """Maps string labels to binary (0/1)."""
    label_str = str(label).lower()
    if 'non-vpn' in label_str: return 0
    if 'vpn' in label_str or 'tor' in label_str: return 1
    return 0

def load_and_process_data():
    """Loads ARFF files and prepares the DataFrame."""
    base_dir = 'ml/data/processed/'
    # Update this list with your actual file names
    file_names = ['TimeBasedFeatures-Dataset-15s-VPN.arff'] 
    
    data_frames = []
    print("[INFO] Loading datasets...")
    
    for fname in file_names:
        path = os.path.join(base_dir, fname)
        if not os.path.exists(path):
            print(f"[WARN] File not found: {path}")
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().replace("''", "?")
            data, meta = arff.loadarff(io.StringIO(content))
            df_part = pd.DataFrame(data)
            for col in df_part.select_dtypes([object]).columns:
                df_part[col] = df_part[col].str.decode('utf-8')
            data_frames.append(df_part)
        except Exception as e:
            print(f"[ERROR] Loading {fname}: {e}")

    if not data_frames:
        raise ValueError("No data loaded. Please check file paths.")

    df = pd.concat(data_frames, ignore_index=True)
    target_col = df.columns[-1]
    
    df['Binary_Label'] = df[target_col].apply(map_label)
    X = df.drop(columns=[target_col, 'Binary_Label'])
    y = df['Binary_Label']
    
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    # Sanitize column names for Boosting models
    clean_feature_names = [re.sub(r'[<>\\[\\]]', '_', name) for name in X.columns]
    X.columns = clean_feature_names
    
    return X, y

def find_best_threshold_cv(estimator, X, y, cv=5):
    """
    Finds the optimal decision threshold using Cross-Validation on the TRAINING set.
    """
    print(f"[INFO] Finding optimal threshold via {cv}-Fold CV on Training Data...")
    y_probas = cross_val_predict(estimator, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
    
    precisions, recalls, thresholds = precision_recall_curve(y, y_probas)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)
    
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"[INFO] Optimal CV Threshold: {best_thresh:.4f} (CV F1: {best_f1:.4f})")
    return best_thresh, best_f1

# ==========================================
# 3. Visualization Functions
# ==========================================
def generate_shap_report(pipeline, X_test, feature_names, output_dir):
    """
    Generates SHAP interpretability report for the XGBoost base model.
    """
    print("[INFO] Generating SHAP report...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # 1. Prepare data: Apply preprocessing (cleaning + selection + quantile transform)
        # pipeline[:-1] retrieves all steps except the final classifier
        preprocessor = pipeline[:-1]
        X_test_transformed = preprocessor.transform(X_test)
        
        # 2. Extract base model: Get fitted XGBoost from StackingClassifier
        stacking_clf = pipeline.named_steps['stacking']
        base_model = None
        
        for model in stacking_clf.estimators_:
            if isinstance(model, xgb.XGBClassifier):
                base_model = model
                break
        
        if base_model is None:
            base_model = stacking_clf.estimators_[0]
            print(f"[WARN] XGBoost not found in stack, using {type(base_model).__name__} for SHAP.")

        # 3. Compute SHAP values
        # Sample 500 instances for speed
        sample_size = min(500, len(X_test_transformed))
        indices = np.random.choice(len(X_test_transformed), sample_size, replace=False)
        X_sample = X_test_transformed[indices]
        
        # Use TreeExplainer (optimized for tree-based models)
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_sample)

        # 4. Plot and save Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title(f'SHAP Summary ({type(base_model).__name__})')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_summary.png")
        plt.close()

        # 5. Plot and save Bar Plot (Feature Importance)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
        plt.title('Feature Importance (SHAP)')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_importance_bar.png")
        plt.close()

        print(f"[INFO] SHAP reports saved to {output_dir}/")

    except Exception as e:
        print(f"[WARN] Failed to generate SHAP report: {e}")
        import traceback
        traceback.print_exc()

def generate_evaluation_plots(y_true, y_probs, y_pred, pipeline, output_dir):
    """Generates and saves standard evaluation plots."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-VPN', 'VPN'], yticklabels=['Non-VPN', 'VPN'])
    plt.title('Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()

    # 2. ROC and PR Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='#ff7f0e', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    
    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    ax2.plot(recall, precision, color='#2ca02c', lw=2, label=f'PR (AUC = {pr_auc:.3f})')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_pr_curves.png")
    plt.close()

    # 3. Meta-Learner Weights
    try:
        stacking_clf = pipeline.named_steps['stacking']
        meta_model = stacking_clf.final_estimator_
        model_names = [est[0] for est in stacking_clf.estimators]
        coeffs = meta_model.coef_[0]
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x=model_names, y=coeffs, palette='viridis')
        plt.title('Meta-Learner Weights (Importance)')
        plt.axhline(0, color='black', linewidth=0.8)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/meta_learner_weights.png")
        plt.close()
    except Exception as e:
        print(f"[WARN] Could not plot meta-learner weights: {e}")

# ==========================================
# 4. Optuna Optimization Wrappers
# ==========================================

def optimize_xgboost(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'objective': 'binary:logistic',
        'n_jobs': -1,
        'random_state': 42
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        model = xgb.XGBClassifier(**params)
        model.fit(X[train_idx], y.iloc[train_idx])
        preds = model.predict(X[val_idx])
        scores.append(f1_score(y.iloc[val_idx], preds))
    return np.mean(scores)

def optimize_lightgbm(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'objective': 'binary',
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        model = lgb.LGBMClassifier(**params)
        model.fit(X[train_idx], y.iloc[train_idx])
        preds = model.predict(X[val_idx])
        scores.append(f1_score(y.iloc[val_idx], preds))
    return np.mean(scores)

def optimize_rf(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 25),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'n_jobs': -1,
        'random_state': 42
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        model = RandomForestClassifier(**params)
        model.fit(X[train_idx], y.iloc[train_idx])
        preds = model.predict(X[val_idx])
        scores.append(f1_score(y.iloc[val_idx], preds))
    return np.mean(scores)

# ==========================================
# 5. Main Pipeline
# ==========================================

def run_pipeline():
    # --- A. Data Loading ---
    X, y = load_and_process_data()
    
    # Stratified Split: 80% Train, 20% Test (Hold-out)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

    # --- B. Preprocessing for Optimization ---
    print("[INFO] Preparing data for hyperparameter search...")
    # Manually run preprocessing to provide clean data to Optuna
    cleaner = VPNFeaturePreprocessor()
    selector = CorrelationSelector(threshold=0.95)
    qt = QuantileTransformer(output_distribution='normal', random_state=42)

    X_train_clean = cleaner.transform(X_train)
    selector.fit(X_train_clean)
    X_train_sel = selector.transform(X_train_clean)
    X_train_opt = qt.fit_transform(X_train_sel) # Returns numpy array
    
    print(f"[INFO] Features reduced from {X.shape[1]} to {X_train_opt.shape[1]}")

    # --- C. Hyperparameter Optimization ---
    N_TRIALS = 15 # Increase this for production (e.g., 50)

    print(f"\n--- Optimizing Models ({N_TRIALS} trials each) ---")
    
    # 1. XGBoost
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(lambda t: optimize_xgboost(t, X_train_opt, y_train), n_trials=N_TRIALS)
    best_xgb = study_xgb.best_params
    best_xgb.update({'objective': 'binary:logistic', 'n_jobs': -1})
    
    # 2. LightGBM
    study_lgbm = optuna.create_study(direction='maximize')
    study_lgbm.optimize(lambda t: optimize_lightgbm(t, X_train_opt, y_train), n_trials=N_TRIALS)
    best_lgbm = study_lgbm.best_params
    best_lgbm.update({'objective': 'binary', 'n_jobs': -1, 'verbose': -1})
    
    # 3. Random Forest
    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(lambda t: optimize_rf(t, X_train_opt, y_train), n_trials=N_TRIALS)
    best_rf = study_rf.best_params
    best_rf.update({'n_jobs': -1, 'random_state': 42})

    print("[INFO] Hyperparameter optimization complete.")

    # --- D. Training Stacking Ensemble ---
    print("\n[INFO] Training Final Stacking Pipeline...")
    
    estimators = [
        ('xgb', xgb.XGBClassifier(**best_xgb)),
        ('lgbm', lgb.LGBMClassifier(**best_lgbm)),
        ('rf', RandomForestClassifier(**best_rf))
    ]
    
    final_estimator = LogisticRegression()

    full_pipeline = Pipeline([
        ('preprocessor', VPNFeaturePreprocessor()),
        ('selector', selector), # Use the fitted selector
        ('quantile_transform', QuantileTransformer(output_distribution='normal', random_state=42)),
        ('stacking', StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            n_jobs=-1,
            passthrough=False
        ))
    ])

    full_pipeline.fit(X_train, y_train)

    # --- E. Threshold Optimization (No Leakage) ---
    # Find best threshold on TRAIN data using CV
    optimal_thresh, cv_f1 = find_best_threshold_cv(full_pipeline, X_train, y_train)

    # --- F. Unified Evaluation (On Memory Test Set) ---
    print("\n" + "="*40)
    print("       FINAL EVALUATION (TEST SET)       ")
    print("="*40)

    # 1. Inference Latency
    start_time = time.time()
    probs_test = full_pipeline.predict_proba(X_test)[:, 1]
    end_time = time.time()
    
    latency_ms = ((end_time - start_time) / len(X_test)) * 1000
    
    # 2. Apply Threshold
    final_preds = (probs_test >= optimal_thresh).astype(int)

    # 3. Metrics
    acc = accuracy_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds)
    roc_auc = roc_auc_score(y_test, probs_test)
    
    print(f"Optimal Threshold: {optimal_thresh:.4f}")
    print(f"Accuracy:          {acc:.4f}")
    print(f"F1 Score:          {f1:.4f}")
    print(f"ROC AUC:           {roc_auc:.4f}")
    print(f"Latency:           {latency_ms:.4f} ms/sample")
    print("-" * 40)
    print("Classification Report:")
    print(classification_report(y_test, final_preds, target_names=['Non-VPN', 'VPN']))
    
    # 4. Generate Plots
    plot_dir = 'evaluation_results'
    generate_evaluation_plots(y_test, probs_test, final_preds, full_pipeline, plot_dir)
    print(f"[INFO] Plots saved to {plot_dir}/")

    final_feature_names = list(X_train_sel.columns)
    generate_shap_report(full_pipeline, X_test, final_feature_names, plot_dir)

    # 5. Threshold Stability Check
    p, r, t = precision_recall_curve(y_test, probs_test)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores_test = 2*(p*r)/(p+r)
    best_test_thresh = t[np.nanargmax(f1_scores_test)]
    
    print("\n--- Stability Check ---")
    print(f"Train Threshold: {optimal_thresh:.4f} | Test Optimal: {best_test_thresh:.4f}")
    if abs(optimal_thresh - best_test_thresh) < 0.1:
        print("[PASS] Threshold is stable.")
    else:
        print("[WARN] Threshold drift detected.")

    # --- G. Save Artifacts ---
    output_dir = 'server'
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    model_artifacts = {
        'pipeline': full_pipeline,
        'threshold': optimal_thresh,
        'feature_names': list(X_train_sel.columns),
        'metrics': {'f1': f1, 'accuracy': acc, 'cv_f1': cv_f1},
        'params': {'xgb': best_xgb, 'lgbm': best_lgbm, 'rf': best_rf}
    }
    
    joblib.dump(model_artifacts, f'{output_dir}/vpn_model_artifacts.pkl')
    print(f"\n[INFO] Model artifacts saved to {output_dir}/vpn_model_artifacts.pkl")

# ==========================================
# 6. Inference Service
# ==========================================

class VPNInferenceService:
    """
    Encapsulates model inference logic for the application.
    Handles loading the model, aligning features, and making predictions.
    """
    def __init__(self, artifacts_path='server/vpn_model_artifacts.pkl'):
        self.artifacts_path = artifacts_path
        self.pipeline = None
        self.threshold = 0.5
        self.feature_names_in = None
        self.is_loaded = False
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            if os.path.exists(self.artifacts_path):
                print(f"[INFO] Loading artifacts from {self.artifacts_path}...")
                # Note: joblib loading requires VPNFeaturePreprocessor and CorrelationSelector
                # to be defined in the current namespace (which they are in this file).
                artifacts = joblib.load(self.artifacts_path)
                
                self.pipeline = artifacts['pipeline']
                self.threshold = artifacts.get('threshold', 0.5)
                
                # Retrieve the feature list from training
                if 'feature_names' in artifacts:
                    self.feature_names_in = artifacts['feature_names']
                elif hasattr(self.pipeline, 'feature_names_in_'):
                    self.feature_names_in = self.pipeline.feature_names_in_
                
                self.is_loaded = True
                print(f"[INFO] Model loaded. Threshold: {self.threshold:.4f}")
            else:
                print(f"[ERROR] Artifact file not found: {self.artifacts_path}")
        except Exception as e:
            print(f"[ERROR] Model load failed: {e}")
            import traceback
            traceback.print_exc()

    def predict(self, X_inference):
        """
        Executes prediction on the input dataframe.
        :param X_inference: Raw input DataFrame (usually from core_logic.extract_features)
        :return: Tuple (vpn_probs, predictions, error_msg)
        """
        if not self.is_loaded or self.pipeline is None:
            return None, None, "Model not loaded."

        try:
            # 1. Clean column names (consistent with training)
            clean_names = [re.sub(r'[<>\\[\\]]', '_', name) for name in X_inference.columns]
            X_inference.columns = clean_names

            # 2. Feature Alignment
            # Ensure the DataFrame column order and quantity match training exactly
            if self.feature_names_in is not None:
                # Fill missing columns with 0
                missing_cols = set(self.feature_names_in) - set(X_inference.columns)
                for c in missing_cols:
                    X_inference[c] = 0
                
                # Remove extra columns and reorder according to training order
                X_inference = X_inference[self.feature_names_in]

            # 3. Predict Probabilities
            probs_all = self.pipeline.predict_proba(X_inference)
            vpn_probs = probs_all[:, 1]

            # 4. Apply Threshold
            predictions = (vpn_probs >= self.threshold).astype(int)

            return vpn_probs, predictions, None

        except Exception as e:
            return None, None, str(e)

if __name__ == "__main__":
    run_pipeline()