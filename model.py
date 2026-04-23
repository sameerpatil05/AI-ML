# ============================================================
#  model.py  —  Employee Salary Predictor
#  Full ML Pipeline: Load → Clean → EDA → Train → Evaluate
# ============================================================

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model   import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

# ── 1. LOAD DATA ────────────────────────────────────────────
print("=" * 55)
print("  EMPLOYEE SALARY PREDICTOR — ML PIPELINE")
print("=" * 55)

df = pd.read_csv("dataset.csv")
print(f"\n[1] Dataset loaded  →  Shape: {df.shape}")

# ── 2. DATA CLEANING ────────────────────────────────────────
print("\n[2] Data Cleaning")
print(f"    Null values:\n{df.isnull().sum().to_string()}")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(f"    After cleaning  →  Shape: {df.shape}")

# ── 3. EDA (console summary) ────────────────────────────────
print("\n[3] Exploratory Data Analysis")
print(df.describe().round(2).to_string())

print("\n    Correlation with Annual_Salary:")
corr = df.corr()["Annual_Salary"].drop("Annual_Salary").sort_values(ascending=False)
print(corr.round(4).to_string())

# ── 4. FEATURE SELECTION ────────────────────────────────────
features = ["Years_Experience", "Age", "Education_Level",
            "Hours_Per_Week", "Num_Projects"]
target   = "Annual_Salary"

X = df[features]
y = df[target]
print(f"\n[4] Features selected: {features}")

# ── 5. TRAIN / TEST SPLIT ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[5] Train/Test split  →  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ── 6. TRAIN MODEL ──────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)
print("\n[6] LinearRegression model trained ✔")

# ── 7. EVALUATE ─────────────────────────────────────────────
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test,  y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\n[7] Model Evaluation on Test Set")
print(f"    MAE   : ${mae:,.2f}")
print(f"    MSE   : ${mse:,.2f}")
print(f"    RMSE  : ${rmse:,.2f}")
print(f"    R²    : {r2:.4f}  ({r2*100:.2f}%)")

print("\n    Feature Coefficients:")
for feat, coef in zip(features, model.coef_):
    print(f"      {feat:<22}: {coef:+.4f}")
print(f"    Intercept             : {model.intercept_:+.4f}")

# ── 8. SAVE MODEL ───────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\n[8] Model saved  →  model.pkl ✔")
print("=" * 55)
