# VPNScope Pro: ML-Based Encrypted Traffic Detection

**VPNScope Pro** is an advanced machine learning framework designed to detect VPN (Virtual Private Network) and Tor usage within network traffic. By analyzing time-based flow statistics, it distinguishes between normal (clear/direct) traffic and encrypted tunnel traffic using a robust **Stacking Ensemble** model.

This project covers the entire pipeline: from exploratory data analysis (EDA) and hyperparameter optimization (Optuna) to model training, evaluation, and a web-based inference interface (Gradio).

## Key Features

  * **Advanced Ensemble Learning:** Utilizes a **Stacking Classifier** combining:
      * **Random Forest:** For stability and handling non-linear data.
      * **XGBoost:** For high-performance gradient boosting.
      * **MLP (Multi-Layer Perceptron):** To capture complex neural patterns.
      * **Logistic Regression:** As the meta-learner to combine predictions.
  * **Automated Optimization:** Implements **Optuna** for hyperparameter tuning to maximize F1-Scores.
  * **Robust Preprocessing:** Includes custom pipeline steps for log-transformation (`Log1p`) and correlation-based feature selection.
  * **Comprehensive Evaluation:** Detailed metric reporting (Accuracy, F1, ROC-AUC, Confusion Matrices).
  * **Interactive Web UI:** A **Gradio** interface to upload `.pcap` files and get instant forensic analysis.

## Project Structure

```text
vpnscope-ml/
├── .gitignore               # Git configuration
├── requirements.txt         # Python dependencies
├── ml/                      # Machine Learning Research & Analysis
│   ├── analysis/            # Scripts for EDA and Evaluation
│   │   ├── 01_data_distribution.py
│   │   ├── 02_pca.py
│   │   ├── 03_label.py
│   │   └── 04_model_evaluation.py
│   ├── data/
│   │   ├── processed/       # ARFF datasets (Time-based features)
│   │   └── raw/             # Raw PCAP files (VPN/Non-VPN)
│   └── vpn_scope.ipynb      # Jupyter Notebook for prototyping
└── server/                  # Production/Inference Code
    ├── app.py               # Gradio Web Application entry point
    ├── core_logic.py        # Feature extraction logic (PCAP parsing)
    ├── model.py             # Main training & optimization pipeline
    └── vpn_model_artifacts.pkl # Saved model (generated after training)
```

## Installation

### Prerequisites

  * Python 3.8+
  * `libpcap` (required for Scapy/traffic analysis)

### Steps

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/astroyee/vpnscope-ml.git
    cd vpnscope-ml
    ```

2.  **Create a Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

## Usage Workflow

### 1\. Data Preparation & Exploration

Before training, you can analyze the dataset distributions and feature correlations using the scripts in `ml/analysis/`.

```bash
# Analyze class distribution and skewness
python ml/analysis/01_data_distribution.py

# Perform dimensionality reduction analysis
python ml/analysis/02_pca.py
```

### 2\. Model Training & Optimization

Run the main training script. This script performs the following:

1.  Loads `.arff` data.
2.  Runs **Optuna** trials to find the best hyperparameters for XGBoost, RF, and MLP.
3.  Trains the Stacking Ensemble on the full dataset.
4.  Finds the optimal probability threshold.
5.  Saves the pipeline to `server/vpn_model_artifacts.pkl`.

<!-- end list -->

```bash
python server/model.py
```

### 3\. Model Evaluation

Generate detailed performance reports, ROC curves, and Confusion Matrix visualizations.

```bash
python ml/analysis/04_model_evaluation.py
```

*Artifacts (images) will be saved to the `evaluation_results/` directory.*

## Launching the Gradio Interface

To start the web-based dashboard for analyzing real traffic files (`.pcap` or `.pcapng`), follow these steps:

### 1\. Start the Server

Run the application script located in the server directory:

```bash
python server/app.py
```

### 2\. Access the UI

Once the server is running, you will see a local URL in your terminal. Open your web browser and navigate to:

`http://127.0.0.1:7860`

From this interface, you can drag and drop packet capture files to perform automated VPN detection analysis.

## Web Interface Features

The **VPNScope Pro** UI allows analysts to:

1.  **Upload:** Drag and drop `.pcap` or `.pcapng` files.
2.  **Analyze:** Automatically extracts flow features (Duration, Inter-arrival time, Bytes/sec, etc.).
3.  **Visualize:**
      * **Verdict Banner:** Color-coded status (VPN/Normal) with a "Suspicion Score".
      * **Flow Table:** Detailed list of suspicious flows sorted by probability.
      * **Protocol Distribution:** Charts showing transport layer stats (TCP/UDP).
      * **Top Talkers:** Source IP traffic volume analysis.

## Disclaimer

This tool is intended for **network research, academic study, and authorized security auditing**. Ensure you have permission to capture and analyze network traffic on the target network. The authors are not responsible for any misuse of this software.

## License

This project is open-source. Please refer to the repository license for usage terms.