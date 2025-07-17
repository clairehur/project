import pandas as pd
import os

# Parameters
horizon = 20
lookback = 60
file_path = f"C:/Users/YourName/Downloads/180_features_selected_t+{horizon}.csv"

# Load data
df = pd.read_csv(file_path, parse_dates=['date'])

# Sort by ric then date for proper time alignment
df = df.sort_values(['ric', 'date'])

# Identify feature columns (exclude non-feature ones)
exclude_cols = ['date', 'ric', f'utilization_t+{horizon}']
feature_cols = [col for col in df.columns if col not in exclude_cols]

# Generate lag features efficiently
lagged_features = []
for lag in range(1, lookback + 1):
    shifted = df.groupby('ric')[feature_cols].shift(lag)
    shifted.columns = [f'{col}_lag{lag}' for col in feature_cols]
    lagged_features.append(shifted)

# Concatenate all lag features
df_lagged = pd.concat([df] + lagged_features, axis=1)

# Drop rows with any NaN (due to lagging)
df_lagged.dropna(inplace=True)

# Final feature and target split
X = df_lagged.drop(columns=exclude_cols)
y = df_lagged[f'utilization_t+{horizon}']

print("✅ Feature matrix shape:", X.shape)
print("✅ Target vector shape:", y.shape)
