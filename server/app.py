import gradio as gr
import pandas as pd
import numpy as np
import os
from model_pipeline import (
    VPNInferenceService, 
    VPNFeaturePreprocessor, 
    CorrelationSelector
)
from core_logic import extract_features, calculate_suspicion_score

# --- 1. Model Initialization ---
# Classes like VPNFeaturePreprocessor are now handled internally by VPNInferenceService
ARTIFACTS_PATH = 'server/vpn_model_artifacts.pkl'
inference_service = VPNInferenceService(ARTIFACTS_PATH)

def analyze_pcap_file(file_obj):
    empty_outputs = (
        _render_status("Waiting", "Please upload a file.", "#9ca3af"),
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    )

    if file_obj is None:
        return empty_outputs

    try:
        print(f"Analyzing: {file_obj.name}")
        
        # 1. Extract features (delegated to core_logic)
        X_inference, df_meta = extract_features(file_obj.name)
        
        if X_inference.empty:
            return (_render_status("Error", "No valid flows found.", "#ef4444"), *empty_outputs[1:])
            
        # 2. Model Inference (delegated to model_pipeline service)
        # Handles feature alignment, probability prediction, and thresholding
        vpn_probs, predictions, error = inference_service.predict(X_inference)

        if error:
            return (_render_status("Error", error, "#ef4444"), *empty_outputs[1:])

        # 3. Calculate Suspicion Score (Business Logic)
        suspicion_score = calculate_suspicion_score(df_meta, vpn_probs)
        
        # 4. Determine Banner display
        vpn_count = np.sum(predictions)
        
        if suspicion_score >= 50.0: 
            banner = _render_status(
                "VPN Traffic Detected", 
                f"Suspicion Score: {suspicion_score:.1f} / 100", 
                "#dc2626", 
                "🚨",
                detail=f"Model flagged {vpn_count} flows as encrypted tunnels."
            )
        else:
            banner = _render_status(
                "Normal Traffic", 
                f"Suspicion Score: {suspicion_score:.1f} / 100", 
                "#16a34a", 
                "✅",
                detail="Traffic patterns appear consistent with standard web usage."
            )

        # 5. Prepare display data
        results_df = df_meta.copy()
        results_df['Type'] = ["🔴VPN" if p == 1 else "✅Normal" for p in predictions]
        results_df['VPN Prob'] = (vpn_probs).round(4)
        
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

# --- Gradio UI Definition ---
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