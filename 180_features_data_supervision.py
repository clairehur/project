import pandas as pd
import numpy as np

# Parameters
horizon = 20
lookback = 60
file_path = f"C:/Users/YourName/Downloads/180_features_selected_t+{horizon}.csv"

# Load and sort
df = pd.read_csv(file_path, parse_dates=['date'])
df.sort_values(['ric', 'date'], inplace=True)

# Identify features to lag
exclude_cols = ['date', 'ric', f'utilization_t+{horizon}']
feature_cols = [col for col in df.columns if col not in exclude_cols]

# Downcast to float32 to save memory
df[feature_cols] = df[feature_cols].astype(np.float32)

# Initialize df_lagged with index and core columns only
df_lagged = df[['date', 'ric', f'utilization_t+{horizon}']].copy()

# Generate lagged features in-place
for lag in range(1, lookback + 1):
    lagged = df.groupby('ric')[feature_cols].shift(lag)
    lagged.columns = [f'{col}_lag{lag}' for col in feature_cols]
    lagged = lagged.astype(np.float32)  # cast early
    df_lagged = pd.concat([df_lagged, lagged], axis=1)
    
    # Optional: periodically drop NaNs to avoid memory buildup
    if lag % 10 == 0:
        df_lagged.dropna(inplace=True)
        df_lagged.reset_index(drop=True, inplace=True)

# Final cleanup
df_lagged.dropna(inplace=True)
df_lagged.reset_index(drop=True, inplace=True)

# Extract final feature and target sets
X = df_lagged.drop(columns=['date', 'ric', f'utilization_t+{horizon}'])
y = df_lagged[f'utilization_t+{horizon}']

print("✅ Memory-efficient lagging complete.")
print("X shape:", X.shape)
print("y shape:", y.shape)



# ===== merge other utilization_t+horizon onto the lagged df ======

next_horizons = [1, 5, 10, 15]

for h in next_horizons:
    target_file = f"C:/Users/YourName/Downloads/target_extracted_{h}.csv"
    
    # Load only necessary columns
    target_df = pd.read_csv(target_file, parse_dates=['date'])
    
    target_col = f'utilization_t+{h}'
    target_df = target_df[['date', 'ric', target_col]]
    
    # Merge target into df_lagged
    df_lagged = df_lagged.merge(target_df, on=['date', 'ric'], how='left')

print("✅ All targets merged. Final shape:", df_lagged.shape)

