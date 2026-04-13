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
