
# Loan Default Prediction Script (Fixed Version with Preprocessing)

# ================================
# 1. Imports
# ================================
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ================================
# 2. Load datasets
# ================================
df1 = pd.read_csv("C:/Users/iamsu/OneDrive/Desktop/train.csv")
df2 = pd.read_csv("C:/Users/iamsu/OneDrive/Desktop/Test.csv")

# Clean column names (lowercase + strip spaces)
df1.columns = df1.columns.str.strip().str.lower()
df2.columns = df2.columns.str.strip().str.lower()

# Use training data
df = df1.copy()

# Drop missing values (consider imputation for better results)
df = df.dropna()

# Ensure output folder exists
os.makedirs("outputs", exist_ok=True)

# ================================
# 3. Target column
# ================================
target_col = "default"
print(f"✅ Using target column: {target_col}")

# ================================
# 4. Features & Target split
# ================================
X = df.drop(columns=[target_col, "loanid"], errors="ignore")
y = df[target_col]

# Auto-detect categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"🔍 Detected {len(categorical_cols)} categorical columns: {categorical_cols}")
print(f"🔍 Detected {len(numerical_cols)} numerical columns: {numerical_cols}")

# ================================
# 5. Train/Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create preprocessor: Encode categoricals + Scale numericals
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # Keep any unexpected columns (if any)
)

# Fit preprocessor on train and transform
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"✅ Preprocessing complete. Train shape: {X_train_processed.shape}, Test shape: {X_test_processed.shape}")

# ================================
# 6. Model Training
# ================================
# Logistic Regression (on preprocessed data)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_processed, y_train)
y_pred_lr = log_reg.predict(X_test_processed)

print("\n📊 Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Random Forest with Grid Search (on preprocessed data)
rf = RandomForestClassifier(random_state=42)
param_grid = {"n_estimators": [100, 200], "max_depth": [5, 10, None]}
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train_processed, y_train)  # Use processed data

best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test_processed)  # Use processed data

print("\n🌲 Random Forest Results:")
print("Best Params:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ================================
# 7. Predictions on Test.csv
# ================================
test_df = df2.copy()

# Ensure test has same columns as train (drop extras, add missing as NaN if needed)
missing_cols = set(X.columns) - set(test_df.columns)
if missing_cols:
    print(f"⚠️ Warning: Test set missing columns: {missing_cols}. Filling with NaN.")
    for col in missing_cols:
        test_df[col] = np.nan

extra_cols = set(test_df.columns) - set(X.columns) - {'loanid'}
if extra_cols:
    print(f"⚠️ Warning: Test set has extra columns: {extra_cols}. Dropping them.")
    test_df = test_df.drop(columns=extra_cols)

# Drop loanid for prediction
X_test_final = test_df.drop(columns=["loanid"], errors="ignore")

# Transform test data with preprocessor (no fitting!)
X_test_final_processed = preprocessor.transform(X_test_final)

# Predict using best RF model
test_preds = best_rf.predict(X_test_final_processed)

# Save predictions
preds_df = pd.DataFrame({
    "loanid": test_df["loanid"],
    "predicted_default": test_preds
})

preds_df.to_csv("outputs/test_predictions.csv", index=False)
print("✅ Predictions saved to outputs/test_predictions.csv")

# ================================
# 8. Confusion Matrix Visualization
# ================================
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

print("📊 Confusion matrix saved to outputs/confusion_matrix.png")
