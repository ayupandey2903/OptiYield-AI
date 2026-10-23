# OptiYield AI

Python-based machine learning and operations engine for predictive manufacturing defect analysis and smart inventory optimization. The framework predicts equipment failure risks from industrial sensor telemetry, applies dynamic safety stock allocation rules, computes optimal inventory reorder parameters, and exports scenario outputs to CSV.

This project was built to demonstrate an end-to-end MLOps workflow, feature engineering pipeline, model evaluation metrics, and practical operational decisions driven by predictive AI.

## Project Overview

The model represents industrial telemetry streams and an automated inventory optimization engine. For each dataset execution, it:

1. Preprocesses sensor telemetry data including temperature, rotational speed, torque, and tool wear.
2. Engineers interaction terms and scales numerical feature distributions.
3. Trains an XGBoost classification model to estimate machine failure probabilities.
4. Evaluates model performance using ROC-AUC, Precision, Recall, and F1-score metrics.
5. Applies probability-driven rules to calculate dynamic safety stock levels (`Recommended_Safety_Stock`).
6. Computes reorder thresholds to mitigate stockouts while controlling holding costs.
7. Validates performance against target thresholds (ROC-AUC > 0.90).
8. Exports detailed recommendations and inventory schedules to CSV.

## Repository Structure
## Core Model Components

`Preprocess`
Models the raw sensor telemetry streams. It handles data transformations using:
* Feature selection and unique ID removal
* Numerical standard scaling (`StandardScaler`)
* Categorical encoding for machine types
* Interaction feature creation (e.g., thermal power metrics)

The output includes clean feature matrices ready for model training and evaluation.

`TrainModel`
Primary machine learning pipeline. It builds and cross-validates the prediction model:
* Fits an `XGBoostClassifier` on failure labels
* Evaluates cross-validation ROC-AUC scores
* Computes feature importance rankings across telemetry sensors

`OptimizeInventory`
Main orchestration class. Translates predicted failure risks into operational inventory decisions.

`SafetyStockCalculator`
Handles dynamic safety stock scaling based on risk tiers:
* **High Failure Risk (> 0.50):** Expands safety stock buffer by 2.5x
* **Moderate Risk (0.20 – 0.50):** Expands safety stock buffer by 1.5x
* **Low Risk (< 0.20):** Maintains base safety stock buffer

## Key Assumptions

* Sensor readings arrive in sequential time-series intervals.
* Failure probability directly correlates with component replacement urgency.
* Base safety stock is set to a default baseline of 50 units.
* Model validation passes when test set ROC-AUC meets or exceeds `0.95`.

## Installation

Create a virtual environment, install the dependencies, and run the pipeline.

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt


Validation
The model includes two validation layers:

Model Performance Validation: Checks whether test set ROC-AUC score exceeds target threshold (0.9539 achieved) and flags results as PASS or CHECK.

Inventory Tie-Out Validation: Verifies that calculated safety stock levels cover expected failure risk windows without exceeding maximum storage capacity limits.

Interpretation
Higher Failure Probabilities (>0.50): Instantly trigger higher safety stock buffers to prevent unplanned downtime.

Lower Probabilities (<0.20): Maintain baseline inventory levels to minimize holding costs.

Feature Importance: Tool wear and thermal readings typically serve as the strongest early-warning indicators of failure.

Disclaimer
This project is for educational and portfolio demonstration purposes only. It is not intended for direct commercial manufacturing execution without independent testing and domain validation.
