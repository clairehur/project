import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import seaborn as sns

def load_data(horizon):
    files = {
        'avail': r'd:\df_feature_files\df_efp_avail_qty.csv',
        'used': r'd:\df_feature_files\df_efp_used_qty.csv',
        'util': r'd:\df_feature_files\df_utilization.csv',
        'total': r'd:\df_feature_files\df_total_avail_qty.csv',
        'total_used': r'd:\df_feature_files\df_total_usage_qty.csv',
        'rba': r'd:\df_feature_files\df_rba_avail_qty.csv',
        'net_qty': r'd:\df_feature_files\df_net_avail_qty.csv',
        'pth_qty': r'd:\df_feature_files\df_pth_qty.csv',
        'usage_qty': r'd:\df_feature_files\df_usage_qty.csv',
        'pred': fr'd:\project_part2\model_one_predicted_{horizon}.csv'  # Assumes pred files are named like this
    }
    return {k: pd.read_csv(v) for k, v in files.items()}

def merge_data(d):
    df = d['avail'].merge(d['used'], on=['date', 'ric'], suffixes=('_avail', '_used'))
    df = df.merge(d['util'], on=['date', 'ric'])
    df = df.merge(d['total'], on=['date', 'ric'])
    df = df.merge(d['total_used'], on=['date', 'ric'])
    df = df.merge(d['rba'], on=['date', 'ric'])
    df = df.merge(d['net_qty'], on=['date', 'ric'])
    df = df.merge(d['pth_qty'], on=['date', 'ric'])
    df = df.merge(d['usage_qty'], on=['date', 'ric'])
    return df

def clean_data(df):
    mask = (df['total_avail_qty'] > df['efp_avail_qty']) & (df['total_usage_qty'] > df['efp_used_qty'])
    total_ratio = df['total_usage_qty'] / df['total_avail_qty']
    efp_ratio = df['efp_used_qty'] / df['efp_avail_qty']
    mask1 = ~((total_ratio == 1) & (efp_ratio != 1))
    mask2 = df['efp_avail_qty'] != 0
    mask3 = ~((df['efp_avail_qty'] / df['total_avail_qty'] == 1) & (df['rba_avail_qty'] != 0))
    df = df[mask & mask1 & mask2 & mask3].copy()
    df['avail_qty'] = df['efp_avail_qty'].replace([np.inf, -np.inf], np.nan)
    df['used_qty'] = df['efp_used_qty'].replace([np.inf, -np.inf], np.nan)
    df['z'] = (df['efp_used_qty'] / df['efp_avail_qty']).replace([np.inf, -np.inf], np.nan).clip(upper=1)
    df['y'] = df['efp_avail_qty'] / df['total_avail_qty']
    df = df.dropna(subset=['avail_qty', 'used_qty', 'z', 'utilization', 'y'])
    return df

def build_mapping(df, n_neighbors=500):
    mapping_data = df[['utilization', 'y', 'z']]
    X = mapping_data[['utilization', 'y']].values
    y = mapping_data['z'].values
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
    knn.fit(X, y)
    return knn

def plot_single_point_multi_horizon(ric, date):
    horizons = [1, 5, 10, 15, 20]
    fig, ax = plt.subplots(figsize=(10, 6))
    pastel_blue = "#A7C7E7"
    pastel_pink = "#F7CAC9"
    
    for horizon in horizons:
        data = load_data(horizon)
        df = merge_data(data)
        df = clean_data(df)
        interp = build_mapping(df)
        
        pred_df = data['pred']
        df_indexed = df[['date', 'ric', 'efp_avail_qty', 'total_avail_qty']].drop_duplicates()
        pred_df = pred_df.merge(df_indexed, on=['date', 'ric'], how='left')
        
        row = pred_df[(pred_df['ric'] == ric) & (pred_df['date'] == date)].dropna(subset=['efp_avail_qty', 'total_avail_qty', 'y_pred'])
        
        if row.empty:
            print(f"No data found for horizon {horizon}, ric={ric}, date={date}")
            continue
        
        row = row.iloc[0]
        x_input = row['y_pred']
        efp_avail_qty = row['efp_avail_qty']
        total_avail_qty = row['total_avail_qty']
        
        scales = np.arange(-0.25, 3.05, 0.1)
        x_scale = np.linspace(-25, 300, num=len(scales))  # Adjusted to percentage change
        y_inputs = (total_avail_qty - efp_avail_qty) + (efp_avail_qty * (1 + scales))  # Adjust for percentage change
        X_inputs = np.column_stack((np.full(len(scales), x_input), y_inputs / total_avail_qty))
        z_estimated = interp.predict(X_inputs)
        
        ax.plot(x_scale, z_estimated, label=f'Horizon {horizon}', linewidth=1.5)
    
    ax.axvline(x=0, color=pastel_pink, linestyle='--', linewidth=1.5)  # Assuming 0% change line
    ax.set_title(f'{ric}\n{date}\nAll Horizons', fontsize=12, fontweight='light')
    ax.set_xlabel('EFP Avail Qty Change (%)', fontsize=11)
    ax.set_ylabel('EFP Usage %', fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(x_scale.min(), x_scale.max())
    ax.grid(False)
    ax.set_facecolor("#F8F8F8")
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.tight_layout()
    plt.legend()
    plt.savefig(fr'D:\project_part2\imp_efp_usage_rate_vs_efp_avail_%_change_model_one_all_horizons_{ric}_{date}_pastel.png', dpi=300)
    plt.show()

# Example call with fixed ric and date
plot_single_point_multi_horizon(ric='603162.SS', date='2025-02-25')