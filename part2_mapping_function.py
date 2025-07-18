import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator

# ======= LOAD FILES =======

def load_data():
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
        'pred': r'd:\project_part2\final_predicted.csv'
    }
    return {k: pd.read_csv(v) for k, v in files.items()}


# ======= MERGE DATA =======

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


# ======= CLEAN DATA =======

def clean_data(df):
    mask = (df['total_avail_qty'] > df['efp_avail_qty']) & (df['total_usage_qty'] > df['efp_used_qty'])
    total_ratio = df['total_usage_qty'] / df['total_avail_qty']
    efp_ratio = df['efp_used_qty'] / df['efp_avail_qty']
    mask1 = ~((total_ratio == 1) & (efp_ratio == 1))
    mask2 = df['efp_avail_qty'] != 0
    mask3 = (((df['efp_avail_qty'] / df['total_avail_qty']) == 1) & (df['rba_avail_qty'] != 0))

    df = df[mask & mask1 & mask2 & mask3].copy()

    df['avail_qty'] = df['efp_avail_qty'].replace([np.inf, -np.inf], np.nan)
    df['used_qty'] = df['efp_used_qty'].replace([np.inf, -np.inf], np.nan)
    df['z'] = (df['efp_used_qty'] / df['efp_avail_qty']).replace([np.inf, -np.inf], np.nan).clip(upper=1)
    df['y'] = df['efp_avail_qty'] / df['total_avail_qty']

    df = df.dropna(subset=['avail_qty', 'used_qty', 'z', 'utilization', 'y'])
    return df


# ======= BUILD MAPPING FUNCTION =======

def build_mapping(df):
    mapping_data = df[['utilization', 'y', 'z']].dropna()
    points = mapping_data[['utilization', 'y']].values
    values = mapping_data['z'].values
    return LinearNDInterpolator(points, values)


# ======= VECTORIZED PREDICT USING MAPPING =======

def predict_z(pred_df, df, interpolator, scales):
    pred_df = pred_df.rename(columns={'y_pred': 'utilization_pred'})

    # Merge efp and total avail qty
    df_indexed = df[['date', 'ric', 'efp_avail_qty', 'total_avail_qty']].drop_duplicates()
    pred_df = pred_df.merge(df_indexed, on=['date', 'ric'], how='left')

    results = []
    for scale in scales:
        y_input = (pred_df['efp_avail_qty'] * scale) / (
            (pred_df['total_avail_qty'] - pred_df['efp_avail_qty']) + (pred_df['efp_avail_qty'] * scale)
        )
        z_estimated = interpolator(pred_df['utilization_pred'], y_input)

        temp = pred_df[['date', 'ric']].copy()
        temp['scale'] = scale
        temp['x'] = pred_df['utilization_pred']
        temp['y_input'] = y_input
        temp['z_estimated'] = z_estimated
        results.append(temp)

    return pd.concat(results, ignore_index=True)


# ======= MAIN RUNNER =======

def main():
    data = load_data()
    df = merge_data(data)
    df = clean_data(df)
    interp = build_mapping(df)

    scales = [0.5, 1, 1.5, 2, 2.5]
    results_df = predict_z(data['pred'], df, interp, scales)
    print(results_df)


if __name__ == '__main__':
    main()
