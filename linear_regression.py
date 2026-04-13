# ================================
# 1. IMPORT LIBRARIES
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================================
# 2. LOAD DATASET
# ================================
df = pd.read_csv("insurance_data_linear.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ================================
# 3. DATA EXPLORATION
# ================================
print("\nMissing Values:")
print(df.isnull().sum())

plt.figure()
sns.histplot(df['charges'], kde=True)
plt.title("Distribution of Medical Charges")
plt.show()

plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ================================
# 4. DATA PREPROCESSING
# ================================
df = pd.get_dummies(df, drop_first=True)

X = df.drop("charges", axis=1)
y = df["charges"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================
# 5. TRAIN-TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ================================
# 6. MODEL TRAINING
# ================================
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ================================
# 7. MODEL EVALUATION
# ================================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Charges")
plt.show()

# ================================
# 8. CROSS VALIDATION (NEW)
# ================================
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')

print("\nCross Validation Scores:", cv_scores)
print("Average CV Score:", np.mean(cv_scores))
