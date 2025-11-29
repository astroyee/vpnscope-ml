import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
import re
from sklearn.base import BaseEstimator, TransformerMixin

# Assume core logic file core_logic.py exists and provides extraction functions
# If core_logic.py also contains relevant class definitions, ensure there are no conflicts
try:
    from core_logic import extract_features, calculate_suspicion_score
except ImportError:
    # Simple mock to prevent UI check failure due to missing core_logic
    def extract_features(f): return pd.DataFrame(), pd.DataFrame()
    def calculate_suspicion_score(m, p): return 0

# --- 1. Must redefine classes here ---
# Joblib needs to find these class definitions in the current namespace when loading the model
# Must match the class definitions in the training code exactly

class VPNFeaturePreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        if isinstance(X_copy, pd.DataFrame):
            num_cols = X_copy.select_dtypes(include=[np.number]).columns
            # Compatible with applymap/map in new and old pandas versions
            func = lambda x: 0 if x < 0 else x
            try:
                X_copy[num_cols] = X_copy[num_cols].map(func)
            except:
                X_copy[num_cols] = X_copy[num_cols].applymap(func)
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

# --- 2. Global Variables and Configuration ---
ARTIFACTS_PATH = 'server/vpn_model_artifacts.pkl'

# Global variables
pipeline = None
optimal_threshold = 0.5
model_features_in = None

def load_artifacts():
    global pipeline, optimal_threshold, model_features_in
    try:
        if os.path.exists(ARTIFACTS_PATH):
            print(f"Loading artifacts from {ARTIFACTS_PATH}...")
            artifacts = joblib.load(ARTIFACTS_PATH)
            
            # Unpack from dictionary
            pipeline = artifacts['pipeline']
            optimal_threshold = artifacts.get('threshold', 0.5)
            
            # Try to get the input feature list from training (Sklearn Pipeline usually has this attribute)
            # If not, rely on feature_names potentially saved in artifacts
            if hasattr(pipeline, 'feature_names_in_'):
                model_features_in = pipeline.feature_names_in_
            
            print(f"Model loaded successfully. Optimal Threshold: {optimal_threshold:.4f}")
            return True
        else:
            print(f"Artifact file not found: {ARTIFACTS_PATH}")
            return False
    except Exception as e:
        print(f"Model load failed: {e}")
        return False

# Initial load
load_artifacts()

def analyze_pcap_file(file_obj):
    empty_outputs = (
        _render_status("Waiting", "Please upload a file.", "#9ca3af"),
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    )

    if file_obj is None:
        return empty_outputs

    try:
        print(f"Analyzing: {file_obj.name}")
        
        # 1. Extract features
        X_inference, df_meta = extract_features(file_obj.name)
        
        if X_inference.empty:
            return (_render_status("Error", "No valid flows found.", "#ef4444"), *empty_outputs[1:])
            
        if not pipeline:
            return (_render_status("Error", "Model not loaded.", "#ef4444"), *empty_outputs[1:])

        # 2. Clean column names (consistent with training)
        clean_names = [re.sub(r'[<>\\[\\]]', '_', name) for name in X_inference.columns]
        X_inference.columns = clean_names

        # 3. Feature Alignment - Very Important!
        # Ensure the DataFrame column order and quantity input to the Pipeline match training exactly
        if model_features_in is not None:
            # Fill missing columns with 0
            missing_cols = set(model_features_in) - set(X_inference.columns)
            for c in missing_cols:
                X_inference[c] = 0
            
            # Remove extra columns and reorder according to training order
            X_inference = X_inference[model_features_in]

        # 4. Predict probabilities
        # Pipeline automatically executes: Preprocessor -> Selector -> Scaler -> Stacking -> Meta Model
        probs_all = pipeline.predict_proba(X_inference)
        vpn_probs = probs_all[:, 1]
        
        # 5. Classify using the optimal threshold (instead of default predict)
        predictions = (vpn_probs >= optimal_threshold).astype(int)

        # 6. Calculate Suspicion Score
        # Can still use the original logic here, or base it directly on the mean of vpn_probs
        suspicion_score = calculate_suspicion_score(df_meta, vpn_probs)
        
        # Determine Banner display
        # Logic: If a certain number of flows are determined to be VPN and the score is high, alert
        # Can keep this business-level threshold, or combine with prediction counts
        if suspicion_score >= 50.0: 
            banner = _render_status(
                "VPN Traffic Detected", 
                f"Suspicion Score: {suspicion_score:.1f} / 100", 
                "#dc2626", 
                "🚨",
                detail=f"Model flagged {np.sum(predictions)} flows as encrypted tunnels."
            )
        else:
            banner = _render_status(
                "Normal Traffic", 
                f"Suspicion Score: {suspicion_score:.1f} / 100", 
                "#16a34a", 
                "✅",
                detail="Traffic patterns appear consistent with standard web usage."
            )

        # 7. Prepare display data
        results_df = df_meta.copy()
        results_df['Type'] = ["🔴VPN" if p == 1 else "✅Normal" for p in predictions]
        results_df['VPN Prob'] = (vpn_probs).round(4)
        
        vpn_count = np.sum(predictions)
        normal_count = len(predictions) - vpn_count
        df_type_counts = pd.DataFrame({
            "Traffic Type": ["VPN / Encrypted", "Normal / Clear"],
            "Count": [vpn_count, normal_count]
        })

        df_proto_counts = results_df['Protocol'].value_counts().reset_index()
        df_proto_counts.columns = ["Protocol", "Count"]

        df_ip_counts = results_df['Source IP'].value_counts().head(10).reset_index()
        df_ip_counts.columns = ["Source IP", "Flow Count"]

        display_table = results_df.sort_values(by="VPN Prob", ascending=False)
        display_table = display_table[["Source IP", "Dest IP", "Protocol", "Length", "Type", "VPN Prob"]]

        return (
            banner, 
            display_table,
            df_type_counts,
            df_proto_counts,
            df_ip_counts
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (_render_status("System Error", str(e), "#ef4444"), *empty_outputs[1:])

def _render_status(title, desc, color, icon="", detail=""):
    return f"""
    <div style="background-color: {color}; padding: 25px; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
        <div style="font-size: 4rem; margin-bottom: 10px;">{icon}</div>
        <h2 style="margin:0; font-size: 2.2rem; font-weight: 700;">{title}</h2>
        <p style="margin: 10px 0 0 0; font-size: 1.5rem; font-weight: 500; opacity: 0.95;">{desc}</p>
        <p style="margin-top: 10px; font-size: 0.9rem; opacity: 0.8; font-style: italic;">{detail}</p>
    </div>
    """

# --- Gradio UI Definition (Keep as is, just fine-tune some descriptions) ---
with gr.Blocks(title="VPNScope Pro") as demo:
    
    gr.Markdown(
        """
        # 🛡️ VPNScope Pro: Intelligent Traffic Analysis (Stacking Ensemble)
        AI-powered forensic analysis using XGBoost + Random Forest + MLP Ensemble.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1, min_width=350):
            upload_file = gr.File(
                label="Step 1: Upload Traffic Dump",
                file_types=[".pcap", ".pcapng"],
                file_count="single"
            )
            analyze_btn = gr.Button("🚀 Analyze Traffic", variant="primary", size="lg")
            
            status_html = gr.HTML(label="Verdict")
            
            gr.Markdown("### 📊 Composition")
            chart_type = gr.BarPlot(
                x="Traffic Type", 
                y="Count", 
                tooltip=["Traffic Type", "Count"],
                y_lim=[0, None],
                height=250,
                show_label=False
            )

        with gr.Column(scale=3):
            with gr.Tabs():
                
                with gr.TabItem("📋 Suspicious Flows"):
                    gr.Markdown("Flows sorted by **VPN Probability**. High probability indicates encrypted tunnels.")
                    detail_table = gr.Dataframe(
                        headers=["Source IP", "Dest IP", "Protocol", "Length", "Type", "VPN Prob"],
                        datatype=["str", "str", "str", "number", "str", "number"],
                        interactive=False,
                    )

                with gr.TabItem("📈 Protocol Stats"):
                    with gr.Row():
                        chart_proto = gr.BarPlot(
                            x="Protocol", 
                            y="Count", 
                            title="Transport Layer",
                            tooltip=["Protocol", "Count"],
                            height=300
                        )
                        chart_ip = gr.BarPlot(
                            x="Source IP", 
                            y="Flow Count", 
                            title="Top Talkers",
                            tooltip=["Source IP", "Flow Count"],
                            height=300,
                            x_label_angle=45
                        )

    analyze_btn.click(
        fn=analyze_pcap_file,
        inputs=upload_file,
        outputs=[
            status_html,
            detail_table,
            chart_type,
            chart_proto,
            chart_ip
        ]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0")