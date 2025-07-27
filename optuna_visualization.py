import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_hex
from matplotlib.cm import ScalarMappable
from pandas.plotting import parallel_coordinates

# Load your CSV file
file_path = r'D:\xgb_ALL_ric_ALL_horizon\impt_optuna_runs\optuna_trials_dataframe_t+20.csv'
df = pd.read_csv(file_path)

# Optional: Filter to completed trials if your CSV has a 'state' column
if 'state' in df.columns:
    df = df[df['state'] == 'COMPLETE']

# Add a 'trial' column for the parallel_coordinates function (unique per row)
df['trial'] = df['number'] if 'number' in df.columns else df.index

# Automatically detect hyperparameter columns (adjust list if needed)
hyperparams = [col for col in df.columns if col.startswith('params_')]

# Create a plotting DataFrame with objective value first
plot_df = df[['trial', 'value'] + hyperparams].copy()
plot_df.rename(columns={'value': 'objective_value'}, inplace=True)  # Rename for clarity in plot
cols = ['objective_value'] + hyperparams  # Axes order: objective first, then params

# Sort by objective_value ascending (lower = better, so best first)
plot_df = plot_df.sort_values('objective_value')

# Compute colors: Darkest blue for best (low value), lighter for worse (high value)
norm = Normalize(vmin=plot_df['objective_value'].min(), vmax=plot_df['objective_value'].max())
colors = [to_hex(plt.cm.Blues_r(norm(val))) for val in plot_df['objective_value']]  # Reversed Blues for dark=low

# Plot the parallel coordinates
fig, ax = plt.subplots(figsize=(12, 6))  # Adjust size as needed
parallel_coordinates(plot_df, class_column='trial', cols=cols, color=colors, axvlines_kwds={'color': 'gray', 'linestyle': '--'}, ax=ax)

# Customize labels and appearance
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_title('Parallel Coordinate Plot')
ax.set_yticks([])  # Hide y-ticks since values are normalized

# Add min/max labels to each axis
for i, col in enumerate(cols):
    min_val = plot_df[col].min()
    max_val = plot_df[col].max()
    ax.text(i, -0.05, f'{min_val:.4g}', ha='center', va='top', fontsize=8, rotation=45)
    ax.text(i, 1.05, f'{max_val:.4g}', ha='center', va='bottom', fontsize=8, rotation=45)

# Add colorbar with label
sm = ScalarMappable(cmap=plt.cm.Blues_r, norm=norm)
plt.colorbar(sm, ax=ax, label='Objective Value (darker = better)')

plt.tight_layout()
plt.show()  # Displays the plot; or use plt.savefig('parallel_plot.png') to save as image