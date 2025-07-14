import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Configuration
pred_dir = r"D:\xgb_ALL_ric_ALL_horizon"
target_dir = r"D:\xgb_ALL_ric_ALL_horizon"  # adjust if target CSVs are elsewhere
key_horizons = [1, 5, 10, 15, 20]

# Mapping of intermediate horizons to surrounding pairs
interpolation_map = {
    (1, 5): [2, 3, 4],
    (5, 10): [6, 7, 8, 9],
    (10, 15): [11, 12, 13, 14],
    (15, 20): [16, 17, 18, 19],
}

# Collect metrics
results = []

for (low, high), inter_horizons in interpolation_map.items():
    # Load predicted values
    pred_low = pd.read_csv(os.path.join(pred_dir, f"model_one_predicted_{low}.csv"))
    pred_high = pd.read_csv(os.path.join(pred_dir, f"model_one_predicted_{high}.csv"))

    # Ensure same rows are used
    merged = pred_low.merge(pred_high, on=["date", "ric"], suffixes=("_low", "_high"))

    # Average predictions
    merged["interpolated_pred"] = (merged["y_pred_low"] + merged["y_pred_high"]) / 2

    for h in inter_horizons:
        # Load actual target values for horizon h
        target_file = os.path.join(target_dir, f"target_extracted_{h}.csv")
        target_df = pd.read_csv(target_file)

        # Align with prediction by date and ric
        merged_target = merged.merge(target_df, on=["date", "ric"])
        y_true = merged_target[f"utilization_t{h}"]
        y_pred = merged_target["interpolated_pred"]

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        results.append({
            "interpolated_horizon": h,
            "rmse": rmse,
            "mae": mae
        })

# Save metrics
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(pred_dir, "interpolated_horizon_metrics.csv"), index=False)
print(results_df)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(results_df["interpolated_horizon"], results_df["rmse"], marker='o', label="RMSE")
plt.plot(results_df["interpolated_horizon"], results_df["mae"], marker='x', label="MAE")
plt.xlabel("Horizon")
plt.ylabel("Error")
plt.title("Interpolated Prediction Performance (Test Set)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(pred_dir, "interpolated_horizon_errors.png"), dpi=300)
plt.show()
