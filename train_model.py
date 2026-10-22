import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# 1. Load the Kaggle training dataset
df = pd.read_csv("train.csv")

# 2. Drop unique identifiers
if 'id' in df.columns:
    df = df.drop(columns=['id'])
if 'Product ID' in df.columns:
    df = df.drop(columns=['Product ID'])

# 3. Feature Engineering
if 'Process temperature [K]' in df.columns and 'Air temperature [K]' in df.columns:
    df['Temp_Delta'] = df['Process temperature [K]'] - df['Air temperature [K]']
if 'Rotational speed [rpm]' in df.columns and 'Torque [Nm]' in df.columns:
   df['Power'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']
# 4. Define Target and Features
target_col = 'Machine failure' if 'Machine failure' in df.columns else df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

categorical_cols = [col for col in X.columns if X[col].dtype == 'object' or X[col].nunique() < 10]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# 5. Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 7. Build Full Pipeline with XGBoost Classifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss'))
])

print("\n🚀 Training XGBoost model on industrial dataset... (this may take a few seconds)")
model_pipeline.fit(X_train, y_train)

# 8. Evaluate Model
y_pred = model_pipeline.predict(X_test)
y_prob = model_pipeline.predict_proba(X_test)[:, 1]

print("\n--- Model Performance Evaluation ---")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")