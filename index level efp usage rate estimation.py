import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate index-level EFP usage rate (value-weighted)
def calculate_efp_usage_rate(avail_qty, used_qty, prices, weights=None):
    if weights is None:
        weights = np.ones(len(prices))  # Equal weight if not provided
    num = np.sum(used_qty * prices * weights)
    den = np.sum(avail_qty * prices * weights)
    return num / den if den != 0 else 0

# Load data for fixed horizon (adapt from your load_data/merge_data/clean_data)
def load_and_prepare_data(horizon=20):
    # Your load/merge/clean logic here (from snapshots)
    files = {
        'avail': r'd:\df_feature_files\df_efp_avail_qty.csv',
        'used': r'd:\df_feature_files\df_efp_used_qty.csv',
        'total': r'd:\df_feature_files\df_total_avail_qty.csv',
        # Add others as needed for merge/clean
        'pred': fr'd:\project_part2\model_one_predicted_{horizon}.csv'  # Pred for this horizon
    }
    data = {k: pd.read_csv(v) for k, v in files.items()}
    
    df = data['avail'].merge(data['used'], on=['date', 'ric'], suffixes=('_avail', '_used'))
    df = df.merge(data['total'], on=['date', 'ric'])
    # Apply your full merge_data and clean_data functions here...
    # For demo, assume df now has 'efp_avail_qty', 'efp_used_qty', 'predicted_utilization' (from pred)
    
    # For index-level: Group/aggregate by index (e.g., filter for CSI 300 stocks)
    # Assuming you have a CSV with CSI 300 constituents, prices, weights
    index_df = pd.read_csv('csi300_constituents.csv')  # Load real: columns 'stock', 'price', 'weight'
    # Merge with df for efp qtys (match on 'ric'/'stock')
    index_df = index_df.merge(df, left_on='stock', right_on='ric')  # Adjust as needed
    
    return index_df[['stock', 'price', 'weight', 'efp_avail_qty', 'efp_used_qty']]  # Key columns

# Fixed horizon
horizon = 20
index_data = load_and_prepare_data(horizon)

# Scaling factors (e.g., 0.5 = -50% change, 1.5 = +50%)
scales = np.arange(0.5, 3.1, 0.25)  # Finer steps: 0.5, 0.75, ..., 3.0
x_changes = (scales - 1) * 100  # % change for x-axis

# Calculate rates for each scale
rates = []
for scale in scales:
    new_avail = index_data['efp_avail_qty'] * scale
    rate = calculate_efp_usage_rate(new_avail, index_data['efp_used_qty'], index_data['price'], index_data['weight'])
    rates.append(rate)

# Plot single line for fixed horizon
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_changes, rates, marker='o', color='blue', label=f'Horizon {horizon}')
ax.axvline(0, color='red', linestyle='--', label='Current (0% Change)')
ax.set_title(f'Index-Level EFP Usage Rate vs. EFP Change (CSI 300, Horizon {horizon})')
ax.set_xlabel('EFP Qty Change (%)')
ax.set_ylabel('EFP Usage Rate')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig('csi300_efp_usage_curve_horizon20.png', dpi=300)
plt.show()