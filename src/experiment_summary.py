# Production Design

## 1. System Objective

The production system converts transaction-level behavioral signals into a calibrated fraud-risk probability and an operational decision.

The system is designed around four principles:

1. Predict fraud risk rather than only a binary class.
2. Calibrate predicted probabilities before using them for decision-making.
3. Select intervention thresholds using business costs.
4. Provide explainable reason codes and continuous monitoring.

---

## 2. End-to-End Architecture

```text
Transaction Event
       |
       v
Schema / Data Validation
       |
       v
Feature Engineering
       |
       v
Fraud Risk Model
       |
       v
Probability Calibration
       |
       v
Cost-Sensitive Decision Policy
       |
       +--------------------+
       |                    |
       v                    v
   Low Risk             High Risk
   Approve            Review / Block
                            |
                            v
                    SHAP Reason Codes
                            |
                            v
                    Analyst Feedback
                            |
                            v
                    Model Monitoring
```

---

## 3. Online Inference

For an online transaction, the inference service would receive features such as:

* transaction amount
* account age
* device age
* transaction velocity
* distance from home
* failed authentication attempts
* merchant risk
* international transaction indicator
* new-device indicator
* time-of-day information

The service would perform the following sequence:

```text
Input validation
      ->
Feature transformation
      ->
Model inference
      ->
Probability calibration
      ->
Decision threshold
      ->
Decision + risk score + reason codes
```

The latency-sensitive path should avoid expensive operations that are not required for every transaction.

SHAP explanations can be generated for transactions that are flagged for review or when explanations are required by downstream systems.

---

## 4. Model Layer

The project evaluates both a Logistic Regression baseline and XGBoost.

The baseline is important because model complexity should only be justified when it provides measurable value.

The current experiments show that Logistic Regression achieves stronger validation ranking performance than XGBoost:

* Logistic Regression validation PR-AUC: 0.3484
* Logistic Regression validation ROC-AUC: 0.8201
* Calibrated XGBoost validation PR-AUC: 0.3066
* Calibrated XGBoost validation ROC-AUC: 0.8128

Therefore, the production architecture should not assume that XGBoost is automatically the preferred model.

A production model-selection process would compare candidate models using:

* PR-AUC
* ROC-AUC
* calibration quality
* operational cost
* latency
* interpretability
* robustness across transaction segments

---

## 5. Probability Calibration

The raw XGBoost output is calibrated using sigmoid calibration.

This is important because the downstream decision system interprets the model output as a risk probability rather than simply as a ranking score.

In the current experiment:

| Model              | Test Brier Score |
| ------------------ | ---------------: |
| Raw XGBoost        |           0.0506 |
| Calibrated XGBoost |           0.0290 |

Lower Brier score indicates better probabilistic accuracy.

The calibration layer should be versioned together with the underlying model because changing the model can change the probability distribution.

---

## 6. Cost-Sensitive Decision Policy

The system separates prediction from decision-making.

The model produces:

```text
P(fraud | transaction)
```

The decision layer then compares that probability with an operational threshold.

The current cost assumptions are:

```text
False positive cost = 5
False negative cost = 100
```

The selected validation threshold is:

```text
0.05
```

This threshold is not selected using the held-out test set.

It is selected using validation data and then evaluated once on the held-out test set.

The resulting held-out test performance is:

* Precision: 17.36%
* Recall: 52.91%
* False positives: 519
* False negatives: 97
* Total cost: 12,295

This separation prevents test-set information from being used to optimize the decision policy.

---

## 7. Cost Sensitivity

The optimal threshold changes when the relative cost of a false negative changes.

Current validation experiment:

| False Negative Cost | Optimal Threshold |
| ------------------: | ----------------: |
|                  50 |              0.10 |
|                 100 |              0.05 |
|                 200 |              0.02 |

This demonstrates that there is no universally optimal fraud threshold.

The appropriate operating point depends on the business objective.

A production implementation should therefore keep decision-policy parameters configurable rather than hard-coding a single threshold into the model.

---

## 8. Explainability

The system uses SHAP to generate local reason codes for the underlying XGBoost risk model.

Example reason-code categories include:

* unusually high transaction velocity
* multiple failed attempts
* large transaction amount
* international activity
* new-device activity
* unusual account/device relationships

The project deliberately distinguishes between:

```text
Raw XGBoost
      |
      +--> SHAP explanation
      |
      v
Risk score

Raw XGBoost
      |
      v
Calibration
      |
      v
Calibrated probability
      |
      v
Decision policy
```

Therefore, SHAP explanations describe the underlying XGBoost risk score rather than pretending to directly explain the calibrated wrapper.

---

## 9. Human Review Loop

High-risk transactions can be routed to a human-review queue.

The review system should capture:

```text
Transaction
Risk probability
Decision
Reason codes
Analyst decision
Confirmed fraud / legitimate
Timestamp
Model version
Policy version
```

Confirmed outcomes can later become labeled training data.

This creates a feedback loop:

```text
Prediction
   ->
Human review
   ->
Confirmed outcome
   ->
Labeled data
   ->
Model evaluation
   ->
Retraining
```

Care must be taken to account for selection bias because only reviewed transactions receive human labels.

---

## 10. Monitoring

A production fraud system should monitor at least four categories.

### 10.1 Data Quality

Monitor:

* missing feature rates
* invalid values
* unexpected categorical values
* feature range violations
* schema changes
* event volume

Example alerts:

```text
transaction_amount missing rate > threshold
device_age_days negative values detected
international field contains unexpected values
```

---

### 10.2 Feature Drift

Monitor distributions of important features over time.

Examples:

* transaction amount
* velocity
* merchant risk
* device age
* international activity
* new-device rate

Distribution changes can be measured using techniques such as:

* Population Stability Index
* Jensen-Shannon divergence
* Wasserstein distance

Drift should trigger investigation rather than automatically trigger retraining.

---

### 10.3 Model Performance

Once delayed fraud labels become available, monitor:

* PR-AUC
* recall
* precision
* false-positive rate
* false-negative rate
* cost per transaction
* fraud capture rate

Because fraud labels can arrive later, monitoring should support delayed ground truth.

---

### 10.4 Calibration

Monitor whether predicted probabilities continue to correspond to observed fraud rates.

Useful metrics include:

* Brier score
* calibration curves
* expected calibration error
* observed fraud rate by risk band

For example:

```text
Predicted risk: 10–25%
Expected observed fraud rate should remain
reasonably close to that range.
```

Large divergence indicates probability degradation even when ranking metrics remain acceptable.

---

## 11. Risk-Band Monitoring

The system can aggregate transactions into risk bands:

```text
<1%
1–5%
5–10%
10–25%
25–50%
>=50%
```

For each band, monitor:

* transaction volume
* observed fraud count
* empirical fraud rate
* average predicted probability

This provides an intuitive business-facing view of model behavior.

---

## 12. Model Versioning

Every production prediction should be associated with:

```text
model_version
calibration_version
feature_version
policy_version
```

This makes it possible to reproduce why a transaction received a particular decision.

Example:

```text
model_version = fraud_xgb_v3
calibration_version = sigmoid_v2
feature_version = transaction_features_v4
policy_version = cost_policy_v5
```

---

## 13. Retraining Strategy

Retraining should not occur simply because a single metric moves.

A retraining workflow could be:

```text
New labeled data
      |
Data validation
      |
Train candidate models
      |
Offline evaluation
      |
Calibration evaluation
      |
Segment robustness evaluation
      |
Cost evaluation
      |
Shadow / controlled deployment
      |
Production promotion
```

The candidate model should only replace the production model if it improves the relevant evaluation criteria without unacceptable degradation in other dimensions.

---

## 14. Deployment Strategy

A safe deployment process would be:

### Stage 1 — Offline evaluation

Compare the candidate model with the current production model.

### Stage 2 — Shadow deployment

Generate predictions without affecting transaction decisions.

### Stage 3 — Controlled rollout

Route a limited fraction of traffic through the candidate system.

### Stage 4 — Full deployment

Promote the candidate after monitoring confirms acceptable behavior.

### Stage 5 — Rollback

Maintain the previous model and policy so that the system can quickly revert if monitoring detects a serious regression.

---

## 15. Key Production Risks

Important risks include:

### Concept drift

Fraud behavior changes over time.

### Adversarial adaptation

Fraudsters may adapt to known detection signals.

### Label delay

Confirmed fraud may only become known after the transaction.

### Class imbalance

Fraud is rare, making accuracy a poor primary metric.

### Threshold instability

A threshold optimized for one business-cost assumption may become inappropriate if operational costs change.

### Feedback-loop bias

If only high-risk transactions are reviewed, the resulting labels may not represent the full transaction population.

### Explainability limitations

SHAP explanations describe model behavior and should not automatically be interpreted as causal explanations.

---

## 16. Synthetic Dataset Limitation

The current project uses a controlled synthetic transaction dataset.

This provides an important advantage for experimentation because the data-generation process contains known nonlinear behavioral relationships.

However, synthetic performance should not be interpreted as evidence of production fraud-detection performance.

A real deployment would require validation on representative real-world data, temporal evaluation, stronger leakage controls, privacy/security review, and monitoring after deployment.

A future extension is to benchmark the pipeline against an appropriate public fraud dataset.

---

## 17. Design Principle

The central design principle is:

> **Separate prediction, probability estimation, decision policy, and explanation.**

This prevents the fraud system from becoming a single opaque classifier.

The resulting architecture is:

```text
Features
   |
   v
Prediction Model
   |
   v
Probability Calibration
   |
   v
Cost-Sensitive Policy
   |
   v
Operational Decision
   |
   v
Reason Codes
   |
   v
Monitoring + Feedback
```

This separation makes the system easier to evaluate, explain, tune, monitor, and deploy.
