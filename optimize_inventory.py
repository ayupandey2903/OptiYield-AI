import pandas as pd
from train_model import model_pipeline, X_test, y_test

# 1. Predict failure probabilities on the test set
failure_probs = model_pipeline.predict_proba(X_test)[:, 1]

# 2. Build an Inventory Decision DataFrame
inventory_df = X_test.copy()
inventory_df['Actual_Failure'] = y_test.values
inventory_df['Failure_Probability'] = failure_probs

# 3. Dynamic Safety Stock & Reorder Point Logic
# Base safety stock unit requirement
base_safety_stock = 50 

def calculate_safety_stock(prob):
    # If the AI predicts high failure probability (> 0.5), increase safety stock buffer
    if prob > 0.5:
        return base_safety_stock * 2.5
    elif prob > 0.2:
        return base_safety_stock * 1.5
    else:
        return base_safety_stock

inventory_df['Recommended_Safety_Stock'] = inventory_df['Failure_Probability'].apply(calculate_safety_stock)

# 4. Save and Summarize Results
inventory_df.to_csv("optimized_inventory_recommendations.csv", index=False)
print("\n--- OptiYield AI: Inventory Optimization Engine ---")
print(f"Processed {len(inventory_df)} components.")
print("Sample Recommended Safety Stock Levels (Units):")
print(inventory_df[['Failure_Probability', 'Recommended_Safety_Stock']].head(10))
print("\n✅ Success! Optimized inventory recommendations saved to 'optimized_inventory_recommendations.csv'.")