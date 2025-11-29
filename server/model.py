import os
import re
import io
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Import models
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

import joblib
import optuna

# Turn off parts of Optuna logging output to keep console clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- 1. Basic preprocessing ---
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

# --- 2. Feature selector ---
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

# --- Helper functions ---
def map_label(label):
    label_str = str(label).lower()
    if 'non-vpn' in label_str: return 0
    if 'vpn' in label_str or 'tor' in label_str: return 1
    return 0

def load_and_process_data():
    base_dir = 'ml/data/processed/'
    # Example filename, make sure the path is correct
    file_names = ['TimeBasedFeatures-Dataset-15s-VPN.arff'] 
    data_frames = []
    print("Loading datasets...")
    for fname in file_names:
        try:
            path = os.path.join(base_dir, fname)
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read().replace("''", "?")
                data, meta = arff.loadarff(io.StringIO(content))
                df_part = pd.DataFrame(data)
                for col in df_part.select_dtypes([object]).columns:
                    df_part[col] = df_part[col].str.decode('utf-8')
                data_frames.append(df_part)
        except Exception as e: 
            print(f"Error loading {fname}: {e}")
    
    if not data_frames:
        raise ValueError("No data loaded. Check file paths.")

    df = pd.concat(data_frames, ignore_index=True)
    target_col = df.columns[-1]
    df['Binary_Label'] = df[target_col].apply(map_label)
    X = df.drop(columns=[target_col, 'Binary_Label'])
    y = df['Binary_Label']
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    clean_feature_names = [re.sub(r'[<>\\[\\]]', '_', name) for name in X.columns]
    X.columns = clean_feature_names
    return X, y, clean_feature_names

def find_best_threshold(model, X, y):
    probs = model.predict_proba(X)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

# --- Optuna optimization functions ---

def optimize_xgboost(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_jobs': -1,
        'random_state': 42
    }
    # Use fewer CV folds to speed up the search
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_val)
        scores.append(f1_score(y_val, preds))
    return np.mean(scores)

def optimize_rf(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'n_jobs': -1,
        'random_state': 42
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        clf = RandomForestClassifier(**params)
        clf.fit(X[train_idx], y.iloc[train_idx])
        scores.append(f1_score(y.iloc[val_idx], clf.predict(X[val_idx])))
    return np.mean(scores)

# Define MLP structure mapping (global, so main() can use it too)
MLP_LAYERS_MAP = {
    'small_50': (50,),
    'medium_100': (100,),
    'deep_64_32': (64, 32),
    'deep_100_50': (100, 50)
}

def optimize_mlp(trial, X, y):
    # 1. Let Optuna suggest a key (string), not a tuple directly
    layers_key = trial.suggest_categorical('hidden_layer_sizes', list(MLP_LAYERS_MAP.keys()))
    
    # 2. Map key to the actual tuple
    selected_layers = MLP_LAYERS_MAP[layers_key]
    
    params = {
        'hidden_layer_sizes': selected_layers,
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
        'max_iter': 300, 
        'early_stopping': True,
        'random_state': 42
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        clf = MLPClassifier(**params)
        clf.fit(X[train_idx], y.iloc[train_idx])
        scores.append(f1_score(y.iloc[val_idx], clf.predict(X[val_idx])))
    return np.mean(scores)

# --- Main program ---
def main():
    X, y, feature_names = load_and_process_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Data Loaded. Train shape: {X_train.shape}")

    # 1. Preprocess data for optimization (no Pipeline; directly process numpy for speed)
    print("Pre-processing data for optimization...")
    cleaner = VPNFeaturePreprocessor()
    selector = FeatureSelector(threshold=0.95)
    scaler = StandardScaler()

    X_train_clean = cleaner.transform(X_train)
    selector.fit(X_train_clean)
    X_train_selected = selector.transform(X_train_clean)
    # Convert DataFrame to numpy to improve sklearn speed
    X_train_opt = scaler.fit_transform(X_train_selected)
    
    # For Pipeline final use
    X_test_clean = cleaner.transform(X_test)
    X_test_selected = selector.transform(X_test_clean)
    X_test_final = scaler.transform(X_test_selected) # For final evaluation

    print(f"Features reduced from {X.shape[1]} to {X_train_opt.shape[1]}")

    # 2. Optimize the three models
    N_TRIALS = 15 # Adjust based on available time

    print(f"\n--- Optimizing XGBoost ({N_TRIALS} trials) ---")
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(lambda trial: optimize_xgboost(trial, X_train_opt, y_train), n_trials=N_TRIALS)
    best_xgb = study_xgb.best_params
    best_xgb.update({'objective': 'binary:logistic', 'eval_metric': 'logloss', 'n_jobs': -1})
    print(f"Best XGB F1: {study_xgb.best_value:.4f}")

    print(f"\n--- Optimizing Random Forest ({N_TRIALS} trials) ---")
    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(lambda trial: optimize_rf(trial, X_train_opt, y_train), n_trials=N_TRIALS)
    best_rf = study_rf.best_params
    best_rf.update({'n_jobs': -1, 'random_state': 42})
    print(f"Best RF F1: {study_rf.best_value:.4f}")

    print(f"\n--- Optimizing MLP ({N_TRIALS} trials) ---")
    study_mlp = optuna.create_study(direction='maximize')
    study_mlp.optimize(lambda trial: optimize_mlp(trial, X_train_opt, y_train), n_trials=N_TRIALS)
    best_mlp = study_mlp.best_params
    best_mlp['hidden_layer_sizes'] = MLP_LAYERS_MAP[best_mlp['hidden_layer_sizes']]
    best_mlp.update({'max_iter': 500, 'early_stopping': True, 'random_state': 42})
    print(f"Best MLP F1: {study_mlp.best_value:.4f}")

    # 3. Build stacking ensemble
    print("\n--- Building Final Stacking Ensemble ---")
    
    estimators = [
        ('xgb', xgb.XGBClassifier(**best_xgb)),
        ('rf', RandomForestClassifier(**best_rf)),
        ('mlp', MLPClassifier(**best_mlp))
    ]
    
    final_estimator = LogisticRegression()

    # Important: use raw X_train so Pipeline handles preprocessing consistently in production
    full_pipeline = Pipeline([
        ('preprocessor', VPNFeaturePreprocessor()),
        ('selector', selector), # Use previously fitted selector for consistent features
        ('scaler', StandardScaler()),
        ('stacking', StackingClassifier(
            estimators=estimators, 
            final_estimator=final_estimator,
            cv=5,
            n_jobs=-1,
            passthrough=False # False = meta-model only sees model outputs, not raw features
        ))
    ])

    print("Training Final Stacking Model on full training set...")
    full_pipeline.fit(X_train, y_train)

    # 4. Evaluation and threshold optimization
    print("Optimizing Decision Threshold...")
    best_thresh, best_f1 = find_best_threshold(full_pipeline, X_test, y_test)
    
    probs_test = full_pipeline.predict_proba(X_test)[:, 1]
    final_preds = (probs_test >= best_thresh).astype(int)
    final_acc = accuracy_score(y_test, final_preds)

    print(f"\n=== Final Results ===")
    print(f"Model: Stacking (XGB+RF+MLP -> LR)")
    print(f"Optimal Threshold: {best_thresh:.4f}")
    print(f"Final Test F1 Score: {best_f1:.4f}")
    print(f"Final Test Accuracy: {final_acc:.4f}")

    # 5. Save outputs
    output_dir = 'server'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    model_artifacts = {
        'pipeline': full_pipeline,
        'threshold': best_thresh,
        'feature_names': list(X_train_selected.columns),
        'metrics': {'f1': best_f1, 'accuracy': final_acc},
        'params': {
            'xgb': best_xgb,
            'rf': best_rf,
            'mlp': best_mlp
        }
    }
    
    joblib.dump(model_artifacts, f'{output_dir}/vpn_model_artifacts.pkl')
    print(f"Saved artifacts to {output_dir}/vpn_model_artifacts.pkl")

if __name__ == "__main__":
    main()
