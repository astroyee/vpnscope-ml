# 🛡️ VPNScope ML

**Intelligent Encrypted Traffic Analysis & Classification System**

VPNScope is a machine learning-based forensic tool designed to detect and classify VPN traffic within network flows. Utilizing a **Stacking Ensemble** architecture (XGBoost, LightGBM, Random Forest), it distinguishes between standard web traffic and encrypted VPN tunnels with high precision and sub-millisecond latency.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Scikit--Learn%20%7C%20XGBoost%20%7C%20LightGBM-orange)
![UI](https://img.shields.io/badge/UI-Gradio-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Key Features

* **High-Performance Stacking Ensemble**: Combines the strengths of Gradient Boosting and Bagging models via a Logistic Regression meta-learner.
* **Real-Time Inference**: Ultra-low latency (**~0.0194 ms/sample**), making it suitable for high-throughput network monitoring.
* **Automated Optimization**: Integrated **Optuna** hyperparameter tuning to maximize F1-score automatically.
* **Smart Preprocessing**: Auto-alignment of features, Log1p transformation for skewed data, and correlation-based feature selection.
* **Interactive UI**: A user-friendly **Gradio** web interface for uploading PCAP files and visualizing traffic composition.

---

## 📊 Model Performance

The current model was trained and evaluated on a dataset of ~19,000 flows (80/20 split). It achieves state-of-the-art results in distinguishing VPN traffic.

| Metric | Value | Description |
| :--- | :--- | :--- |
| **ROC AUC** | **0.9810** | Excellent discrimination capability. |
| **F1 Score** | **0.9317** | Balanced precision and recall. |
| **Accuracy** | **92.75%** | Overall correctness on the test set. |
| **Latency** | **0.019 ms** | Average inference time per flow sample. |

### Classification Report (Test Set)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Non-VPN** | 0.94 | 0.91 | 0.92 | 1793 |
| **VPN** | 0.92 | 0.95 | 0.93 | 1959 |

> **Note:** The model uses an optimized decision threshold of **0.4636** to maximize the F1 score, ensuring robustness against class imbalances.

---

## 🏗️ System Architecture

### 1. Data Pipeline
* **Input**: Time-based statistical features (15s duration) extracted from PCAP/ARFF files.
* **Cleaning**: Negative value clipping and missing value handling.
* **Transformation**: `Log1p` transformation to compress feature magnitude and `QuantileTransformer` for normalization.
* **Feature Selection**: Automatically drops highly correlated features (>0.95), reducing dimensionality (e.g., 23 → 15 features).

### 2. The Model (Stacking Classifier)
The system uses a **Stacking Classifier** where a meta-learner aggregates predictions from base models optimized via Optuna:
* **Base Learners**:
    * `XGBoost` (Gradient Boosting)
    * `LightGBM` (Light Gradient Boosting)
    * `RandomForest` (Bagging)
* **Meta Learner**: `LogisticRegression`
* **Optimization**: 5-Fold Cross-Validation on the training set.

---

## 📂 Project Structure

```text
vpnscope-ml/
├── ml/
│   ├── data/               # Raw PCAP and processed ARFF datasets
│   └── analysis/           # Notebooks and scripts for EDA
├── server/
│   ├── app.py              # Gradio Web UI entry point
│   ├── model_pipeline.py   # Training, evaluation, and inference service
│   ├── core_logic.py       # PCAP feature extraction logic
│   └── vpn_model_artifacts.pkl # Saved model pipeline (generated after training)
├── requirements.txt        # Python dependencies
└── README.md
```

## ⚡ Quick Start

### 1\. Installation

Clone the repository and install dependencies:

```bash
git clone [https://github.com/yourusername/vpnscope-ml.git](https://github.com/yourusername/vpnscope-ml.git)
cd vpnscope-ml
pip install -r requirements.txt
```

### 2\. Train the Model

Run the pipeline to preprocess data, optimize hyperparameters, train the ensemble, and save artifacts:

```bash
python server/model_pipeline.py
```

*Output artifacts will be saved to `server/vpn_model_artifacts.pkl`.*

### 3\. Run the UI

Launch the Gradio dashboard to analyze PCAP files:

```bash
python server/app.py
```

Open your browser at `http://localhost:7860`.

-----

## 🖼️ UI Screenshot



The interface provides:

  * **Traffic Verdict**: Immediate visual alert (VPN vs. Normal).
  * **Suspicion Score**: A 0-100 score based on model probability.
  * **Flow Analysis**: Detailed table of suspicious flows (IPs, Protocol, Probability).
  * **Visualizations**: Traffic composition and protocol distribution charts.

-----

## 📝 License

This project is licensed under the MIT License.