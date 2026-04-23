# model.py — Employee Salary Predictor (INR)

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

# 1. Load Data
df = pd.read_csv("dataset.csv")
print(f"Dataset loaded — Shape: {df.shape}")

# 2. Clean Data
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(f"After cleaning — Shape: {df.shape}")

# 3. Features & Target
features = ["Years_Experience", "Age", "Education_Level", "Hours_Per_Week", "Num_Projects"]
target   = "Annual_Salary_INR"

X = df[features]
y = df[target]

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model
model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained ✔")

# 6. Evaluate
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"\nMAE  : ₹{mae:,.0f}")
print(f"MSE  : ₹{mse:,.0f}")
print(f"RMSE : ₹{rmse:,.0f}")
print(f"R2   : {r2:.4f} ({r2*100:.2f}%)")

# 7. Save Model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nModel saved → model.pkl")
