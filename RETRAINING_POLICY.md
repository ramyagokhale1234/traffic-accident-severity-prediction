# Model Retraining Policy

## Traffic Accident Severity Prediction

The deployed accident severity model is continuously monitored for
data quality, class distribution, location drift, model performance,
false-negative rates, and API latency.

## Retraining Criteria

The model should be retrained when one or more of the following
conditions are observed:

### 1. Model Performance Degradation

Retraining is triggered when Macro-F1 decreases by more than 10%
compared with the baseline test performance.

Baseline Macro-F1:

**0.3945**

### 2. Severe Accident Recall

The Fatal and Serious classes are safety-critical.

Retraining should be considered if recall for these classes decreases
significantly from the established baseline.

### 3. Feature Drift

Population Stability Index (PSI) is used to monitor important input
features.

| PSI Value | Interpretation |
|-----------|----------------|
| < 0.10 | No significant drift |
| 0.10–0.20 | Moderate drift |
| > 0.20 | Significant drift — investigate/retrain |

### 4. Data Quality

Retraining or data investigation is required when more than 2% of
incoming records contain missing, invalid, or out-of-range values.

### 5. Prediction Latency

The API should be investigated if prediction latency consistently
exceeds **500 ms**.

### 6. New Training Data

The model should be periodically retrained when a sufficiently large
amount of newly labelled accident data becomes available.

## Retraining Workflow

```text
New Accident Data
        ↓
Data Quality Check
        ↓
Drift Detection
        ↓
Performance Evaluation
        ↓
Retraining Criteria Met?
      /       \
    No         Yes
    ↓           ↓
Continue     Retrain Model
Monitoring       ↓
             MLflow Run
                 ↓
             Evaluate Model
                 ↓
          Register New Version
                 ↓
             Deploy Model