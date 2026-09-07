# Explainable Fraud Detection & Risk Prediction Using Machine Learning

An end-to-end fraud-risk modeling system that combines **imbalanced classification, probability calibration, cost-sensitive decisioning, SHAP explainability, error analysis, and robustness evaluation**.

> **Important:** This project uses a controlled synthetic transaction dataset for reproducible experimentation. The reported metrics should not be interpreted as production fraud-detection performance.

---

## 1. Project Overview

Fraud detection is not simply a binary classification problem.

A useful fraud-risk system should answer four questions:

1. **How likely is this transaction to be fraudulent?**
2. **Can the predicted probability be trusted?**
3. **What action should be taken given the cost of errors?**
4. **Why did the model assign this risk?**

This project addresses these questions through the following pipeline:

```text
Transaction
     |
     v
Feature Engineering
     |
     v
Candidate Models
     |
     v
Probability Calibration
     |
     v
Cost-Sensitive Threshold
     |
     v
Operational Decision
     |
     v
SHAP Reason Codes
     |
     v
Error + Robustness Analysis
     |
     v
Monitoring Design
```

---

# 2. Key Results

The final evaluated calibrated XGBoost model achieved on the **held-out test set**:

| Metric                  |     Result |
| ----------------------- | ---------: |
| PR-AUC                  | **0.2781** |
| ROC-AUC                 | **0.8188** |
| Brier Score             | **0.0290** |
| Fraud prevalence        |  **3.43%** |
| Cost-selected threshold |   **0.05** |
| Precision @ 0.05        | **17.36%** |
| Recall @ 0.05           | **52.91%** |
| False positives         |    **519** |
| False negatives         |     **97** |
| Total cost              | **12,295** |

Because fraud prevalence is only 3.43%, PR-AUC is emphasized over accuracy as a primary ranking metric.

The PR-AUC of 0.2781 is approximately **8.1× the prevalence baseline**.

---

# 3. Model Comparison

A Logistic Regression model was used as a baseline.

| Model               | Validation PR-AUC | Validation ROC-AUC |
| ------------------- | ----------------: | -----------------: |
| Logistic Regression |        **0.3484** |         **0.8201** |
| Raw XGBoost         |            0.2893 |             0.8022 |
| Calibrated XGBoost  |            0.3066 |             0.8128 |

An important result is that **Logistic Regression outperformed XGBoost on validation ranking metrics**.

This result is intentionally retained rather than selecting a more complex model simply because it appears more sophisticated.

Model selection should consider:

* ranking performance
* calibration
* cost
* robustness
* latency
* interpretability

rather than model complexity alone.

---

# 4. Probability Calibration

Raw tree-model probabilities can be poorly calibrated.

The project therefore applies **sigmoid probability calibration** to XGBoost.

### Brier Score

| Model              | Validation | Held-out Test |
| ------------------ | ---------: | ------------: |
| Raw XGBoost        |     0.0503 |        0.0506 |
| Calibrated XGBoost | **0.0283** |    **0.0290** |

The held-out test Brier score decreased from **0.0506 to 0.0290**, demonstrating substantially improved probabilistic accuracy.

Calibration is evaluated independently from ranking performance because a model can rank transactions well while producing poorly calibrated probabilities.

---

# 5. Cost-Sensitive Decisioning

The system separates **prediction** from **decision policy**.

The model estimates:

```text
P(fraud | transaction)
```

The decision layer then determines whether the transaction should be flagged.

The current cost assumptions are:

```text
False positive cost = 5
False negative cost = 100
```

The threshold is selected on the **validation set only**.

The selected threshold was:

```text
0.05
```

It was then locked and evaluated on the untouched test set.

### Validation threshold experiment

| Threshold |  Precision |     Recall | Total Cost |
| --------: | ---------: | ---------: | ---------: |
|      0.01 |      3.45% |    100.00% |     28,965 |
|      0.02 |      8.34% |     73.91% |     13,805 |
|      0.03 |     11.90% |     60.87% |     12,765 |
|  **0.05** | **17.88%** | **54.59%** | **11,995** |
|      0.10 |     27.22% |     45.89% |     12,470 |
|      0.20 |     37.63% |     33.82% |     14,280 |
|      0.50 |     74.19% |     11.11% |     18,440 |

This demonstrates the trade-off between fraud capture and investigation/false-positive cost.

---

# 6. Cost Sensitivity

The optimal threshold changes when the relative cost of missing fraud changes.

| False Negative Cost | Optimal Validation Threshold |
| ------------------: | ---------------------------: |
|                  50 |                     **0.10** |
|                 100 |                     **0.05** |
|                 200 |                     **0.02** |

As the cost of missing fraud increases, the system lowers the intervention threshold and accepts more false positives.

This demonstrates that the decision threshold is a **business-policy parameter**, not an intrinsic property of the classifier.

---

# 7. Explainability with SHAP

The project generates local SHAP reason codes for individual transactions.

Example high-risk signals include:

* failed authentication attempts
* high transaction velocity
* large transaction amount
* international activity
* new-device activity

Example protective signals can include:

* low transaction amount
* low velocity
* small geographic distance

SHAP is applied to the underlying XGBoost model.

The project deliberately distinguishes:

```text
Raw XGBoost
      |
      +----> SHAP explanation
      |
      v
Risk score

Raw XGBoost
      |
      v
Probability calibration
      |
      v
Calibrated probability
      |
      v
Decision
```

Therefore, SHAP explanations describe the underlying XGBoost risk score rather than pretending to directly explain the calibrated wrapper.

---

# 8. Error Analysis

At the cost-selected threshold of 0.05 on the held-out test set:

```text
True positives  = 109
True negatives  = 5,275
False positives = 519
False negatives = 97
```

### False negatives

False-negative transactions had relatively weak observable fraud signals:

* Average transaction amount: **78.09**
* Average 24h velocity: **5.40**
* International rate: **19.6%**
* New-device rate: **22.7%**
* Average failed attempts: **0.44**

This suggests that some fraud events are difficult to distinguish from normal behavior using the available features.

### False positives

False-positive transactions were legitimate transactions with unusual behavior:

* Average transaction amount: **110.75**
* Average 24h velocity: **6.34**
* International rate: **24.9%**
* New-device rate: **23.1%**
* Average failed attempts: **0.67**

This highlights the operational cost of aggressive fraud detection.

---

# 9. Risk-Band Analysis

The empirical fraud rate increases substantially across predicted-risk bands:

| Risk Band | Observed Fraud Rate |
| --------- | ------------------: |
| 1–5%      |               1.81% |
| 5–10%     |               8.07% |
| 10–25%    |              17.10% |
| 25–50%    |              27.20% |
| ≥50%      |              76.00% |

This indicates that the model's risk ranking contains useful information even though probability calibration is not perfect across every band.

---

# 10. Robustness Analysis

The model was evaluated across multiple transaction segments.

### Stronger segments

| Segment          |     PR-AUC |     Recall |
| ---------------- | ---------: | ---------: |
| New device       | **0.3636** | **67.65%** |
| High amount ≥100 | **0.3628** | **63.49%** |
| High velocity ≥5 | **0.3248** | **61.54%** |

### Weaker segments

| Segment                  |     PR-AUC |     Recall |
| ------------------------ | ---------: | ---------: |
| Low velocity <5          | **0.1471** | **26.00%** |
| High merchant risk ≥0.25 | **0.2331** | **52.00%** |
| Existing device          | **0.2546** | **45.65%** |

The analysis is intentionally reported by segment rather than relying only on aggregate metrics.

A production system would monitor such segments for performance degradation and potential distribution shift.

---

# 11. Feature Engineering

The project creates behavioral features including:

* `amount_per_account_day`
* `velocity_ratio`
* `device_account_age_ratio`
* `high_velocity_flag`
* `large_amount_flag`

The transaction-amount threshold is computed using **training data only** to avoid leakage.

---

# 12. Data Generation

The synthetic dataset contains behavioral transaction features such as:

* customer age
* account age
* transaction amount
* 1-hour velocity
* 24-hour velocity
* distance from home
* failed attempts
* device age
* merchant risk
* international activity
* new-device activity

The fraud-generation process includes nonlinear relationships and feature interactions.

The requested fraud prevalence is approximately 3.5%.

The current generated dataset contains:

```text
30,000 transactions
Fraud rate ≈ 3.43%
```

A fixed random seed makes the experiment reproducible.

---

# 13. Experimental Methodology

The project uses:

```text
60% training
20% validation
20% held-out test
```

The split is stratified by fraud label.

The workflow is:

```text
Training
    |
    +--> Model fitting
    |
    +--> Feature-threshold fitting

Validation
    |
    +--> Model comparison
    +--> Calibration evaluation
    +--> Threshold selection
    +--> Cost sensitivity

Held-out Test
    |
    +--> Final performance
    +--> Calibration
    +--> Error analysis
    +--> Robustness
```

The test set is not used to optimize the decision threshold.

---

# 14. Project Structure

```text
explainable-fraud-detection-risk-prediction/
│
├── config.yaml
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── .gitkeep
│
├── reports/
│   ├── experiment_summary.csv
│   ├── calibration_curve.png
│   ├── calibration_results.csv
│   ├── robustness_analysis.csv
│   ├── threshold_analysis.csv
│   ├── cost_sensitivity.csv
│   ├── false_positives.csv
│   ├── false_negatives.csv
│   └── risk_band_analysis.csv
│
├── src/
│   ├── data_generation.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── explain.py
│   ├── error_analysis.py
│   ├── calibration_analysis.py
│   ├── robustness_analysis.py
│   ├── threshold_analysis.py
│   ├── cost_sensitivity.py
│
├── tests/
│   └── test_project.py
│
├── notebooks/
│   └── README.md
│
└── docs/
    └── production_design.md
```

---

# 15. Reproducing the Experiment

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the dataset:

```bash
python -m src.data_generation
```

Train the models:

```bash
python -m src.train
```

Evaluate the final model:

```bash
python -m src.evaluate
```

Run explainability:

```bash
python -m src.explain
```

Run error analysis:

```bash
python -m src.error_analysis
```

Run calibration analysis:

```bash
python -m src.calibration_analysis
```

Run robustness analysis:

```bash
python -m src.robustness_analysis
```

Run threshold analysis:

```bash
python -m src.threshold_analysis
```

Run cost sensitivity:

```bash
python -m src.cost_sensitivity
```

Generate the experiment summary:

```bash
python -m src.experiment_summary
```

Run tests:

```bash
python -m pytest -q
```

---

# 16. Production Design

A production implementation would separate:

```text
Feature validation
        ↓
Feature computation
        ↓
Risk model
        ↓
Probability calibration
        ↓
Decision policy
        ↓
Review / approve / block
        ↓
Reason codes
        ↓
Monitoring
```

Important production monitoring dimensions include:

* data quality
* feature drift
* delayed-label performance
* PR-AUC
* precision
* recall
* false-positive rate
* false-negative rate
* business cost
* calibration
* risk-band stability

The detailed production architecture is documented in:

```text
docs/production_design.md
```

---

# 17. Limitations

This project has several deliberate limitations.

### Synthetic data

The dataset is synthetic and therefore cannot establish real-world fraud-detection performance.

### Static evaluation

The current evaluation is not a temporal production backtest.

### Simplified costs

The false-positive and false-negative costs are illustrative assumptions.

### Limited feature set

Real fraud systems can use much richer behavioral, device, merchant, network, and historical features.

### Delayed labels

Real fraud labels may arrive long after the original transaction.

### Model adaptation

Fraudsters can adapt to detection systems, creating distribution and concept drift.

These limitations are part of the reason the project focuses on **methodology and experimental design**, not claims of production readiness.

---

# 18. Future Work

Potential extensions include:

1. Benchmark against a public real-world fraud dataset.
2. Add temporal train/test evaluation.
3. Compare additional calibration methods.
4. Add expected calibration error.
5. Evaluate temporal and feature drift.
6. Add hyperparameter optimization with nested validation.
7. Add probability/risk-band monitoring dashboards.
8. Add model-version and policy-version tracking.
9. Investigate fairness and subgroup stability where appropriate data is available.
10. Deploy a lightweight inference API for demonstration.

---

# 19. Technical Takeaways

This project demonstrates several applied machine-learning principles:

* **PR-AUC is more informative than accuracy for rare-event detection.**
* **A strong baseline should be retained even when a more complex model is available.**
* **Probability calibration matters when probabilities drive business decisions.**
* **Threshold selection should be separated from model training.**
* **Decision thresholds depend on the relative cost of errors.**
* **Held-out test data should remain untouched during policy optimization.**
* **Error analysis reveals failure modes hidden by aggregate metrics.**
* **Segment-level evaluation can expose robustness weaknesses.**
* **SHAP provides local model explanations but should not be interpreted as causal evidence.**
* **Production ML requires monitoring and feedback loops, not just a trained model.**

---

# 20. Author

Built as an applied machine-learning portfolio project focused on:

**Fraud Detection • Risk Modeling • Explainable AI • Calibration • Cost-Sensitive Learning • Model Evaluation • Robustness • ML System Design**
