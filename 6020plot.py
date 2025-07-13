import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load utilization data
df_util = pd.read_csv("df_utilization.csv", parse_dates=['date'])
df_util.sort_values(['ric', 'date'], inplace=True)

# Create test set (last 20% of dates)
unique_dates = df_util['date'].sort_values().unique()
cutoff_index = int(len(unique_dates) * 0.8)
test_dates_set = set(unique_dates[cutoff_index:])

# Pre-filter test data
df_test = df_util[df_util['date'].isin(test_dates_set)]

# Prepare predictions dictionary
horizons = [1, 5, 10, 15, 20]
pred_map = {}
for h in horizons:
    pred_file = f"model_one_predicted_{h}.csv"
    pred_df = pd.read_csv(pred_file, parse_dates=['date'])
    pred_map[h] = pred_df.set_index(['ric', 'date'])['prediction']

# Generator to yield valid (ric, date) samples
def valid_samples(df, buffer=20):
    for ric in df['ric'].unique():
        df_ric = df_util[df_util['ric'] == ric]
        dates = df_ric['date'].values
        for i in range(len(dates)):
            if dates[i] in test_dates_set and i + buffer < len(dates):
                yield {'ric': ric, 'date': dates[i]}

# Generate 10 random valid samples
valid_df = pd.DataFrame(valid_samples(df_test))
samples = valid_df.sample(n=10).reset_index(drop=True)

# Setup subplots
fig, axes = plt.subplots(5, 2, figsize=(16, 20))  # 5 rows x 2 cols
axes = axes.flatten()

# Loop through samples and plot
for idx, row in samples.iterrows():
    ric = row['ric']
    target_date = row['date']
    ax = axes[idx]

    # Slice window
    df_ric = df_util[df_util['ric'] == ric]
    mask = (df_ric['date'] >= target_date - pd.Timedelta(days=60)) & \
           (df_ric['date'] <= target_date + pd.Timedelta(days=20))
    plot_df = df_ric.loc[mask]

    # Plot actuals
    ax.plot(plot_df['date'], plot_df['utilization'], color='blue', marker='o', label='Actual', markersize=3)
    ax.axvline(target_date, color='gray', linestyle='--', label='T')

    # Plot predictions
    y_T = plot_df.loc[plot_df['date'] == target_date, 'utilization'].values
    if y_T.size:
        y_T = y_T[0]
        for h in horizons:
            future_date = target_date + pd.Timedelta(days=h)
            try:
                pred = pred_map[h].loc[(ric, target_date)]
                ax.plot(future_date, pred, 'ro', markersize=4)
                ax.plot([target_date, future_date], [y_T, pred], 'r--', alpha=0.4)
            except KeyError:
                continue

    # Styling
    ax.set_title(f"{ric} | T = {target_date.strftime('%Y-%m-%d')}")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)

# Final layout
plt.tight_layout()
plt.show()