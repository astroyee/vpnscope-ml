# VPNScope Pro

**ML-Based Encrypted Traffic Detection Tool**

Based on the research paper *"VPNScope Pro: ML-Based Encrypted Traffic Detection Tool"*, this repository implements a Stacking Ensemble machine learning system designed to detect and classify VPN and encrypted tunnel traffic (e.g., Tor) within network flows.

## 📂 Project Structure

  * **`ml/analysis/`**: Scripts for data auditing, statistical analysis, and model benchmarking.
  * **`ml/data/`**: Contains the processed dataset (`.arff`).
  * **`server/`**: Contains the production deployment code and the Gradio web application.

## 🛠️ Installation

1.  **Prerequisites**: Python 3.8+
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage & Execution

Follow these steps to reproduce the analysis, training, and deployment described in the paper.

### 1\. Data Analysis & Preprocessing

Perform a comprehensive audit of the dataset to check for anomalies (negative values, skewness) and generate visualization plots.

  * **Script**: `ml/analysis/data_report.py`
  * **Command**:
    ```bash
    python ml/analysis/data_report.py
    ```
  * **Outputs**:
      * `evaluation_results/analysis_report.txt`: Detailed statistical health check.
      * `evaluation_results/correlation_heatmap.png`: Feature correlation matrix.
      * `evaluation_results/pca_scree_plot.png`: PCA explained variance plot.

### 2\. Baseline Evaluation (Decision Tree)

Run the traditional Decision Tree classifier to establish a performance baseline (\~89% accuracy).

  * **Script**: `ml/analysis/traditional_method.py`
  * **Command**:
    ```bash
    python ml/analysis/traditional_method.py
    ```
  * **Outputs**:
      * Console prints for **Accuracy**, **Confusion Matrix**, and **Classification Report**.

### 3\. Model Training & Comparison (Stacking Ensemble)

Train, tune (using Optuna), and evaluate the Stacking Ensemble (XGBoost, LightGBM, Random Forest) against individual models.

  * **Script**: `ml/analysis/model_report.py`
  * **Command**:
    ```bash
    python ml/analysis/model_report.py
    ```
  * **Outputs**:
      * `tuned_model_comparison_results.csv`: A CSV file comparing Accuracy, F1-Score, and Inference Time for all models.
      * Console logs showing best hyperparameters and final metrics.

### 4\. Run Web Application

Launch the **Gradio** web interface to upload `.pcap` files and perform real-time traffic analysis using the trained model and the **Suspicion Score Engine**.

  * **Script**: `server/app.py`
  * **Command**:
    ```bash
    python server/app.py
    ```
  * **Action**: Open `http://localhost:7860` in your web browser.

## 📊 Key Results

As validated in the accompanying paper:

  * **Selected Model**: Stacking Ensemble (3 Base Learners: XGBoost, LightGBM, RF).
  * **Performance**: 92.70% Accuracy, 0.931 F1-Score.
  * **Latency**: Sub-millisecond inference suitable for real-time monitoring.
