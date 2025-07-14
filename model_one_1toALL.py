import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Directories
pred_dir = r"D:\xgb_ALL_ric_ALL_horizon"
target_dir = r"D:\xgb_ALL_ric_ALL_horizon"

# Load predictions for key horizons
preds = {}
for h in [1, 5, 10, 15, 20]:
    preds[h] = pd.read_csv(os.path.join(pred_dir, f"model_one_predicted_{h}.csv"))
    preds[h].rename(columns={'y_pred': f'y_pred_{h}', 'y_true': f'y_true_{h}'}, inplace=True)

# Merge all predictions on ['date', 'ric']
merged = preds[1]
for h in [5, 10, 15, 20]:
    merged = merged.merge(preds[h], on=['date', 'ric'])

# Result list
results = []

### --- Evaluate Key Horizons (1, 5, 10, 15, 20) ---
for h in [1, 5, 10, 15, 20]:
    y_true = merged[f'y_true_{h}']
    y_pred = merged[f'y_pred_{h}']
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    results.append({
        'interpolated_horizon': h,
        'rmse': rmse,
        'mae': mae
    })

### --- Estimate Intermediate Horizons (Idea 2) ---

# 2–4: avg between 1 and 5
for h in [2, 3, 4]:
    merged[f'y_pred_{h}'] = (merged['y_pred_5'] - merged['y_pred_1']) / 4 + merged['y_pred_1']

# 6–9: avg(1–10)*10 - avg(1–5)*5 / 5
for h in [6, 7, 8, 9]:
    merged[f'y_pred_{h}'] = ((10 * merged['y_pred_10']) - (5 * merged['y_pred_5'])) / 5

# 11–14: avg(1–15)*15 - avg(1–10)*10 / 5
for h in [11, 12, 13, 14]:
    merged[f'y_pred_{h}'] = ((15 * merged['y_pred_15']) - (10 * merged['y_pred_10'])) / 5

# 16–19: avg(1–20)*20 - avg(1–15)*15 / 5
for h in [16, 17, 18, 19]:
    merged[f'y_pred_{h}'] = ((20 * merged['y_pred_20']) - (15 * merged['y_pred_15'])) / 5

### --- Evaluate Interpolated Horizons ---
for h in [2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]:
    target_path = os.path.join(target_dir, f"target_extracted_{h}.csv")
    if not os.path.exists(target_path):
        print(f"Target file not found for T+{h}, skipping.")
        continue

    target_df = pd.read_csv(target_path)
    target_df.rename(columns={f'utilization_t{h}': 'y_true'}, inplace=True)

    # Merge with predicted
    merged_eval = merged.merge(target_df, on=['date', 'ric'])
    y_true = merged_eval['y_true']
    y_pred = merged_eval[f'y_pred_{h}']

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    results.append({
        'interpolated_horizon': h,
        'rmse': rmse,
        'mae': mae
    })

### --- Save and Plot ---
results_df = pd.DataFrame(results).sort_values("interpolated_horizon")
results_df.to_csv(os.path.join(pred_dir, "interpolated_horizon_metrics_idea2_with_keypoints.csv"), index=False)
print(results_df)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(results_df["interpolated_horizon"], results_df["rmse"], marker='o', label="RMSE")
plt.plot(results_df["interpolated_horizon"], results_df["mae"], marker='x', label="MAE")
plt.xlabel("Horizon")
plt.ylabel("Error")
plt.title("Prediction Errors Across All Horizons (Key + Interpolated)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(pred_dir, "interpolated_horizon_errors_idea2_with_keypoints.png"), dpi=300)
plt.show()
