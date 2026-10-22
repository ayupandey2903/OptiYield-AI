# OptiYield AI: Predictive Manufacturing Defect Analysis & Smart Inventory Optimization

OptiYield AI is an end-to-end industrial machine learning and operations framework designed to predict machine failure risks from telemetry sensor data and dynamically calculate safety stock inventory levels.

## 📌 Features & Highlights
* **Predictive Maintenance Model:** Trains an XGBoost classifier on industrial sensor telemetry (temperatures, rotational speed, torque, tool wear) to predict failure events.
* **Exceptional Performance:** Achieved an **ROC-AUC Score of 0.9539** with high precision and recall on unseen evaluation data.
* **Operational Inventory Optimization:** Translates failure probabilities directly into actionable business inventory policies (dynamic safety stock buffering) to minimize stockouts and holding costs.
* **Modular MLOps Architecture:** Structured, clean pipeline separating data preprocessing, model training, and business logic execution.

---

## 🛠 Tech Stack
* **Language:** Python 3
* **Data Processing & Feature Engineering:** `pandas`, `numpy`, `scikit-learn`
* **Machine Learning Model:** `xgboost`
* **Version Control:** Git & GitHub

---

## 📁 Repository Structure

---

## 📊 Model Evaluation
* **ROC-AUC Score:** `0.9539`
* **Target Classification:** Binary Machine Failure Risk Assessment

---

## ⚙️ How to Run

1. **Install Dependencies:**
   ```bash
   pip install pandas scikit-learn xgboost
   python train_model.py
   python optimize_inventory.py
