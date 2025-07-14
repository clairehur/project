import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Directories
pred_dir = r"D:\xgb_ALL_ric_ALL_horizon"
target_dir = r"D:\xgb_ALL_ric_ALL_horizon"

# Load all needed predictions
preds = {}
for h in [1, 5, 10, 15, 20]:
    preds[h] = pd.read_csv(os.path.join(pred_dir, f"model_one_predicted_{h}.csv"))
    preds[h].rename(columns={'y_pred': f'y_pred_{h}'}, inplace=True)

# Merge all predictions on ['date', 'ric']
merged = preds[1]
for h in [5, 10, 15, 20]:
    merged = merged.merge(preds[h], on=['date', 'ric'])

# List of results
results = []

# Estimate for 2, 3, 4: use (y_pred_5 - y_pred_1) / 4
for h in [2, 3, 4]:
    merged[f'y_pred_{h}'] = (merged['y_pred_5'] - merged['y_pred_1']) / 4 + merged['y_pred_1']

# Estimate for 6, 7, 8, 9: ([10×y_pred_10] - [5×y_pred_5]) / 5
for h in [6, 7, 8, 9]:
    merged[f'y_pred_{h}'] = ((10 * merged['y_pred_10']) - (5 * merged['y_pred_5'])) / 5

# Estimate for 11, 12, 13, 14: ([15×y_pred_15] - [10×y_pred_10]) / 5
for h in [11, 12, 13, 14]:
    merged[f'y_pred_{h}'] = ((15 * merged['y_pred_15']) - (10 * merged['y_pred_10'])) / 5

# Estimate for 16, 17, 18, 19: ([20×y_pred_20] - [15×y_pred_15]) / 5
for h in [16, 17, 18, 19]:
    merged[f'y_pred_{h}'] = ((20 * merged['y_pred_20']) - (15 * merged['y_pred_15'])) / 5

# Evaluation loop for all interpolated horizons
for h in [2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]:
    target_file = os.path.join(target_dir, f"target_extracted_{h}.csv")
    if not os.path.exists(target_file):
        print(f"Missing target file for T+{h}: {target_file}")
        continue

    target_df = pd.read_csv(target_file)
    target_df = target_df.rename(columns={f'utilization_t{h}': 'y_true'})

    # Merge with predictions
    eval_df = merged.merge(target_df, on=['date', 'ric'])
    y_true = eval_df['y_true']
    y_pred = eval_df[f'y_pred_{h}']

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    results.append({
        'interpolated_horizon': h,
        'rmse': rmse,
        'mae': mae
    })

# Save results
results_df = pd.DataFrame(results).sort_values('interpolated_horizon')
results_df.to_csv(os.path.join(pred_dir, "interpolated_horizon_metrics_idea2.csv"), index=False)
print(results_df)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(results_df["interpolated_horizon"], results_df["rmse"], marker='o', label="RMSE")
plt.plot(results_df["interpolated_horizon"], results_df["mae"], marker='x', label="MAE")
plt.xlabel("Interpolated Horizon")
plt.ylabel("Error")
plt.title("Idea 2: Interpolated Prediction Errors (Test Set)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(pred_dir, "interpolated_horizon_errors_idea2.png"), dpi=300)
plt.show()
