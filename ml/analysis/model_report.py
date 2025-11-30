import os
import re
import io
import time
import joblib
import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from scipy.io import arff

# Scikit-learn imports
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    precision_score, recall_score
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Constants
RANDOM_STATE = 42
N_JOBS = -1
DATA_PATH = 'ml/data/processed/TimeBasedFeatures-Dataset-15s-VPN.arff' 
N_TRIALS = 10  # You can increase this back to 15 or 20

# ==========================================
# 1. Feature Engineering Classes
# ==========================================

class VPNFeaturePreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if isinstance(X_copy, pd.DataFrame):
            num_cols = X_copy.select_dtypes(include=[np.number]).columns
            X_copy[num_cols] = X_copy[num_cols].apply(lambda x: np.where(x < 0, 0, x))
            X_copy[num_cols] = np.log1p(X_copy[num_cols])
        else:
            X_copy[X_copy < 0] = 0
            X_copy = np.log1p(X_copy)
        return X_copy

class CorrelationSelector(BaseEstimator, TransformerMixin):
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
# 2. Data Loading & Helpers
# ==========================================

def map_label(label):
    label_str = str(label).lower()
    if 'non-vpn' in label_str: return 0
    if 'vpn' in label_str or 'tor' in label_str: return 1
    return 0

def load_and_process_data(file_path):
    print(f"[INFO] Loading data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().replace("''", "?")
    
    data, meta = arff.loadarff(io.StringIO(content))
    df = pd.DataFrame(data)
    
    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].str.decode('utf-8')

    target_col = df.columns[-1]
    df['Binary_Label'] = df[target_col].apply(map_label)
    
    X = df.drop(columns=[target_col, 'Binary_Label'])
    y = df['Binary_Label']
    
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    clean_feature_names = [re.sub(r'[<>\\[\\]]', '_', name) for name in X.columns]
    X.columns = clean_feature_names
    
    return X, y

def convert_optuna_mlp_params(params):
    """
    FIX: Converts Optuna's flat params (n_layers, n_units_l0...) 
    into a valid 'hidden_layer_sizes' tuple for MLPClassifier.
    """
    valid_params = {k: v for k, v in params.items() if not k.startswith('n_layers') and not k.startswith('n_units')}
    
    # Reconstruct hidden_layer_sizes tuple
    if 'n_layers' in params:
        n_layers = params['n_layers']
        layers = []
        for i in range(n_layers):
            # We must retrieve the specific unit count for this layer
            layers.append(params.get(f'n_units_l{i}', 64)) # default to 64 if missing
        valid_params['hidden_layer_sizes'] = tuple(layers)
    
    return valid_params

# ==========================================
# 3. Optuna Optimization Functions
# ==========================================

def create_search_pipeline(model):
    return Pipeline([
        ('preprocessor', VPNFeaturePreprocessor()),
        ('selector', CorrelationSelector(threshold=0.95)),
        ('scaler', StandardScaler()),
        ('clf', model)
    ])

def tune_xgboost(X, y):
    print(f"\n🔧 Tuning XGBoost ({N_TRIALS} trials)...")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'objective': 'binary:logistic',
            'n_jobs': N_JOBS, 'random_state': RANDOM_STATE, 
            'eval_metric': 'logloss', 'verbosity': 0
        }
        pipeline = create_search_pipeline(xgb.XGBClassifier(**params))
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        return cross_val_score(pipeline, X, y, cv=cv, scoring='f1', n_jobs=N_JOBS).mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params

def tune_lightgbm(X, y):
    print(f"\n🔧 Tuning LightGBM ({N_TRIALS} trials)...")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'objective': 'binary',
            'n_jobs': N_JOBS, 'random_state': RANDOM_STATE, 'verbose': -1
        }
        pipeline = create_search_pipeline(lgb.LGBMClassifier(**params))
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        return cross_val_score(pipeline, X, y, cv=cv, scoring='f1', n_jobs=N_JOBS).mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params

def tune_random_forest(X, y):
    print(f"\n🔧 Tuning Random Forest ({N_TRIALS} trials)...")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            'n_jobs': N_JOBS, 'random_state': RANDOM_STATE
        }
        pipeline = create_search_pipeline(RandomForestClassifier(**params))
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        return cross_val_score(pipeline, X, y, cv=cv, scoring='f1', n_jobs=N_JOBS).mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params

def tune_logistic_regression(X, y):
    print(f"\n🔧 Tuning Logistic Regression ({N_TRIALS} trials)...")
    def objective(trial):
        C = trial.suggest_float('C', 0.01, 10.0, log=True)
        params = {
            'C': C,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1000,
            'n_jobs': N_JOBS, 'random_state': RANDOM_STATE
        }
        pipeline = create_search_pipeline(LogisticRegression(**params))
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        return cross_val_score(pipeline, X, y, cv=cv, scoring='f1', n_jobs=N_JOBS).mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params

def tune_mlp(X, y):
    print(f"\n🔧 Tuning MLP ({max(5, N_TRIALS // 2)} trials)...")
    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 3)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f'n_units_l{i}', 32, 128))
        
        params = {
            'hidden_layer_sizes': tuple(layers),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
            'max_iter': 300,
            'random_state': RANDOM_STATE
        }
        pipeline = create_search_pipeline(MLPClassifier(**params))
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        return cross_val_score(pipeline, X, y, cv=cv, scoring='f1', n_jobs=N_JOBS).mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=max(5, N_TRIALS // 2))
    return study.best_params

# ==========================================
# 4. Main Execution
# ==========================================

def get_metrics(name, y_true, y_pred, y_prob):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob) if y_prob is not None else "N/A",
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred)
    }

def main():
    # 1. Load Data
    try:
        X, y = load_and_process_data(DATA_PATH)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # 2. Split Data
    print("[INFO] Splitting data (Train 70% / Test 30%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    # 3. Tuning Phase
    print("\n" + "="*40)
    print("🚀 STEP 1: Hyperparameter Tuning (Optuna)")
    print("="*40)
    
    best_params_xgb = tune_xgboost(X_train, y_train)
    best_params_lgb = tune_lightgbm(X_train, y_train)
    best_params_rf = tune_random_forest(X_train, y_train)
    best_params_lr = tune_logistic_regression(X_train, y_train)
    
    # 4. MLP Tuning & Cleaning (The Fix)
    raw_mlp_params = tune_mlp(X_train, y_train)
    # Convert Optuna format (n_layers, n_units_l0...) to Sklearn format (hidden_layer_sizes=(...))
    best_params_mlp = convert_optuna_mlp_params(raw_mlp_params)
    
    print(f"\n[DEBUG] Converted MLP Params: {best_params_mlp}")

    # 5. Instantiate Optimized Models
    xgb_opt = xgb.XGBClassifier(**best_params_xgb, eval_metric='logloss', n_jobs=N_JOBS, random_state=RANDOM_STATE)
    lgb_opt = lgb.LGBMClassifier(**best_params_lgb, n_jobs=N_JOBS, verbose=-1, random_state=RANDOM_STATE)
    rf_opt = RandomForestClassifier(**best_params_rf, n_jobs=N_JOBS, random_state=RANDOM_STATE)
    lr_opt = LogisticRegression(**best_params_lr, solver='lbfgs', max_iter=1000, n_jobs=N_JOBS, random_state=RANDOM_STATE)
    
    # MLP Initialization using the CLEANED params
    mlp_opt = MLPClassifier(**best_params_mlp, max_iter=300, random_state=RANDOM_STATE)

    # 6. Build Stacking Ensembles
    print("\n[INFO] Building Stacking Ensembles with Optimized Base Learners...")
    
    estimators_3 = [('xgb', xgb_opt), ('lgb', lgb_opt), ('rf', rf_opt)]
    estimators_5 = estimators_3 + [('lr', lr_opt), ('mlp', mlp_opt)]

    # Stacking 1: 3 Models (Meta: Optimized LR)
    stack_3_opt = StackingClassifier(
        estimators=estimators_3, 
        final_estimator=lr_opt, 
        cv=3, n_jobs=N_JOBS
    )

    # Stacking 2: 5 Models (Meta: Optimized LR)
    stack_5_opt = StackingClassifier(
        estimators=estimators_5, 
        final_estimator=lr_opt, 
        cv=3, n_jobs=N_JOBS
    )

    # Stacking 3: 5 Models (Meta: Optimized XGBoost)
    stack_5_xgb_meta = StackingClassifier(
        estimators=estimators_5, 
        final_estimator=xgb_opt,
        cv=3, n_jobs=N_JOBS
    )

    # 7. Evaluation
    models_to_test = [
        ("XGBoost (Tuned)", xgb_opt),
        ("LightGBM (Tuned)", lgb_opt),
        ("RandomForest (Tuned)", rf_opt),
        ("Linear Regression (Tuned)", lr_opt),
        ("MLP (Tuned)", mlp_opt),
        ("Stacking (3 Models | Meta:LR)", stack_3_opt),
        ("Stacking (5 Models | Meta:LR)", stack_5_opt),
        ("Stacking (5 Models | Meta:XGB)", stack_5_xgb_meta)
    ]

    results = []
    print("\n" + "="*40)
    print("⚔️ STEP 2: Final Evaluation on Test Set")
    print("="*40)

    for name, model in models_to_test:
        print(f"Running: {name}...")
        start_time = time.time()
        
        full_pipeline = create_search_pipeline(model)
        full_pipeline.fit(X_train, y_train)
        
        y_pred = full_pipeline.predict(X_test)
        try:
            y_prob = full_pipeline.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_prob = None
            
        metrics = get_metrics(name, y_test, y_pred, y_prob)
        metrics['Time(s)'] = round(time.time() - start_time, 2)
        results.append(metrics)

    # 8. Reporting
    results_df = pd.DataFrame(results)
    cols = ['Model', 'Accuracy', 'F1-Score', 'ROC-AUC', 'Precision', 'Recall', 'Time(s)']
    results_df = results_df[cols].sort_values(by='F1-Score', ascending=False)
    
    print("\n" + "="*60)
    print("🏆 FINAL TUNED MODEL COMPARISON RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    
    output_file = 'tuned_model_comparison_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to {output_file}")

if __name__ == "__main__":
    main()