import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. Load the Kaggle training dataset
try:
    df = pd.read_csv("train.csv")
    print("✅ Successfully loaded 'train.csv'!")
except FileNotFoundError:
    print("❌ Error: 'train.csv' not found in your folder.")
    exit()

# 2. Inspect data columns
print("\nDataset Shape:", df.shape)
print("Columns available:", df.columns.tolist()[:8], "... (truncated)")

# Drop unique identifier columns that don't help prediction
if 'id' in df.columns:
    df = df.drop(columns=['id'])
if 'Product ID' in df.columns:
    df = df.drop(columns=['Product ID'])

# 3. Feature Engineering: Create useful operational metrics
if 'Process temperature [K]' in df.columns and 'Air temperature [K]' in df.columns:
    df['Temp_Delta'] = df['Process temperature [K]'] - df['Air temperature [K]']
if 'Rotational speed [rpm]' in df.columns and 'Torque [Nm]' in df.columns:
    df['Power'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']

# 4. Define Target (Machine Failure) and Features
target_col = 'Machine failure' if 'Machine failure' in df.columns else df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical and numerical columns
categorical_cols = [col for col in X.columns if X[col].dtype == 'object' or X[col].nunique() < 10]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# 5. Build Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

# 6. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit and transform datasets
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\n✅ Data preprocessing & feature engineering complete!")
print("Training feature shape:", X_train_processed.shape)
print("Testing feature shape:", X_test_processed.shape)