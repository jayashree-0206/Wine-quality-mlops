import pandas as pd

data = pd.read_csv("data/winequality-red.csv", sep=";")

print("Dataset shape:", data.shape)
print("\nColumns:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())

print("\nMissing values:")
print(data.isnull().sum())

print("\nQuality distribution:")
print(data["quality"].value_counts().sort_index())