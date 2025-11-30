# 🛡️ VPNScope Pro: Intelligent Traffic Analysis System

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

## 1\. Executive Summary

VPNScope Pro is a machine learning-based forensic tool designed to detect VPN and encrypted tunnel traffic (e.g., Tor) within network capture files (`.pcap`). Unlike signature-based detection (which fails against encrypted payloads), this system utilizes **Time-Based Feature Engineering** and a **Stacking Ensemble Model** to classify traffic flows based on statistical behaviors (inter-arrival times, burstiness, and idle durations).

The final model achieves **92.70% Accuracy** and an **F1-Score of 0.931**, significantly outperforming baseline linear models.

-----

## 2\. Phase I: Data Analysis & Quality Assurance

*Source Script: `ml/analysis/data_report.py`*

Before modeling, a rigorous audit of the dataset (`TimeBasedFeatures-Dataset-15s-VPN.arff`) was conducted to ensure data integrity and understand feature distributions.

### 2.1 Data Health Check

  * **Dataset Volume:** 18,758 traffic flows.
  * **Class Balance:** The dataset is well-balanced (VPN: 52.2%, Non-VPN: 47.8%).
      * **Why this matters:** Balanced data prevents the "accuracy paradox" where a model simply guesses the majority class. No synthetic oversampling (SMOTE) was required.
  * **Data Integrity Issues Detected:**
      * **Negative Values:** The audit revealed negative values in columns like `min_idle` and `min_active`.
      * **Skewness:** 23 out of 23 features had a skewness magnitude \> 1.
      * **Decision:** Since time-based features cannot physically be negative, we implemented a `VPNFeaturePreprocessor` in the pipeline to clip negative values to 0 and apply a `Log1p` transform to correct the heavy right-skew.

### 2.2 Feature Redundancy (PCA & Correlation)

  * **Correlation Analysis:** A heatmap analysis revealed 7 features with correlation \> 0.95 (e.g., `min_active` vs `mean_active`).
      * **Decision:** We implemented a `CorrelationSelector` to automatically drop these redundant features.
      * **Reasoning:** Multicollinearity inflates the variance of coefficient estimates and makes the model unstable, especially for linear meta-learners.
  * **PCA (Principal Component Analysis):** The scree plot indicated that **11 components** explain 99% of the variance.
      * **Insight:** The effective dimensionality of the problem is much lower than the raw feature count (24), confirming that feature selection would be effective without losing information.

-----

## 3\. Phase II: Model Selection & Optimization

*Source Script: `ml/analysis/model_report.py`*

We benchmarked five algorithms and three ensemble strategies. Hyperparameters were tuned using **Optuna** (Bayesian Optimization) to find the global optima efficiently.

### 3.1 Benchmark Results (Test Set)

| Model | Accuracy | F1-Score | ROC-AUC | Inference Time (s) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stacking (3 Models \| Meta:LR)** | **0.9270** | **0.9310** | **0.9808** | **6.01** | **✅ SELECTED** |
| Stacking (5 Models \| Meta:LR) | 0.9270 | 0.9308 | 0.9811 | 29.91 | Too Slow |
| XGBoost (Tuned) | 0.9254 | 0.9295 | 0.9804 | 0.46 | Strong Backup |
| LightGBM (Tuned) | 0.9245 | 0.9288 | 0.9802 | 0.52 | Fast |
| RandomForest (Tuned) | 0.9197 | 0.9245 | 0.9771 | 0.73 | Robust |
| Linear Regression | 0.6326 | 0.6599 | 0.6808 | 1.09 | Failed (Non-linear problem) |

### 3.2 Selection Rationale

1.  **Winner: Stacking Ensemble (3 Models)**
      * **Composition:** XGBoost + LightGBM + Random Forest.
      * **Meta-Learner:** Logistic Regression.
      * **Why:** It provided the highest F1-Score (0.931). While the 5-model stack had similar accuracy, it was **5x slower** (29.91s vs 6.01s). The 3-model stack offers the best trade-off between forensic accuracy and throughput.
2.  **Why Stacking?**
      * Individual tree models (XGBoost, RF) make different types of errors. A stacking classifier learns to weigh the output of these "base learners" to correct their individual biases, resulting in a more robust decision boundary.

-----

## 4\. Phase III: Production Pipeline Architecture

*Source Script: `server/model_pipeline.py`*

The final training pipeline (`run_pipeline`) is engineered for robustness and reproducibility.

### 4.1 Pipeline Steps

1.  **Preprocessing (`VPNFeaturePreprocessor`)**:
      * **Clipping:** Forces all values $\ge 0$.
      * **Log1p Transform:** $x' = \ln(x + 1)$. This compresses the range of highly skewed features (like `flowBytesPerSecond`), making patterns easier for the model to learn.
2.  **Feature Selection (`CorrelationSelector`)**:
      * Removes features with Pearson correlation \> 0.95 to reduce noise.
3.  **Normalization (`QuantileTransformer`)**:
      * Transforms features to follow a Gaussian (Normal) distribution.
      * **Why:** Essential for the Meta-Learner (Logistic Regression) and helps tree-based models converge faster.
4.  **Ensemble Classification (`StackingClassifier`)**:
      * Combines the probability outputs of the pre-trained XGBoost, LightGBM, and Random Forest models.
5.  **Dynamic Thresholding**:
      * Instead of using the default 0.5 threshold, the pipeline runs Cross-Validation to find the **Optimal Threshold** that maximizes the F1-Score.
      * **Result:** This ensures the model is calibrated specifically for the VPN detection task, balancing Precision and Recall.

-----

## 5\. Phase IV: Service Deployment & UI

*Source Script: `server/app.py` & `server/core_logic.py`*

The system is deployed as a web application using **Gradio**, chosen for its ability to rapidly wrap Python ML logic into a user-friendly interface.

### 5.1 System Data Flow

1.  **User Upload:** User uploads a `.pcap` or `.pcapng` file.
2.  **Feature Extraction (`core_logic.py`):**
      * Uses **Scapy** to parse packets.
      * **Logic:** Reconstructs TCP/UDP flows by grouping packets by `(SrcIP, SrcPort, DstIP, DstPort, Proto)`.
      * **Extraction:** Calculates statistical metrics (IAT, Duration, Active/Idle times) matching the training data schema.
3.  **Inference Service (`VPNInferenceService`):**
      * Loads the saved `vpn_model_artifacts.pkl`.
      * Aligns features (handling missing/extra columns) to ensure the input matrix matches the training matrix exactly.
      * Predicts probabilities using the Stacking Ensemble.
4.  **Business Logic Layer:**
      * **Suspicion Score:** We don't just return "VPN" or "Normal." We calculate a weighted score (0-100) based on flow length and protocol (UDP is weighted higher due to VPNs often favoring UDP for speed).
      * **Thresholding:** If Score \> 50, a "Red Alert" banner is displayed.

-----

## 6\. How to Run

### Prerequisites

  * Python 3.8+
  * Wireshark/TShark (for PCAP handling libraries)

### Installation

```bash
pip install -r requirements.txt
```

### 1\. Reproduce Analysis & Training

To generate the data report and retrain the model from scratch:

```bash
# Generate Data Analysis Report
python ml/analysis/data_report.py

# Train Model & Generate Artifacts
python server/model_pipeline.py
```

*Output:* Artifacts will be saved to `server/vpn_model_artifacts.pkl`.

### 2\. Start the Server

To launch the VPNScope Pro web interface:

```bash
python server/app.py
```

Access the UI at `http://localhost:7860`.

### 3\. Usage

1.  Open the web interface.
2.  Upload a `.pcap` file.
3.  Click **Analyze Traffic**.
4.  View the "Suspicion Score", Protocol Stats, and detailed flow table.

-----

## 7\. Artifacts & References

  * **Evaluation Plots:** Saved in `evaluation_results/` (ROC Curves, Confusion Matrix, SHAP plots).
  * **Tuned Results:** `tuned_model_comparison_results.csv`.
  * **Model File:** `server/vpn_model_artifacts.pkl` (Contains the full Stacking Pipeline + Thresholds).