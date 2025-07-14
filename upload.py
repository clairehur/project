from pathlib import Path

# Prepare the full code by combining logic from the user's images with added metric-saving logic

code_string = """
import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Horizon to run
horizons = [1]

# Parameters for XGBoost
params = {
    'max_depth': 6,
    'subsample': 0.8,
    'n_estimators': 2000,
    'eta': 0.2,
    'reg_alpha': 10,
    'reg_lambda': 20,
    'min_child_weight': 5,
    'colsample_bytree': 0.7,
    'random_state': 42,
    'objective': 'multi:softmax'
}

# Directories
download_dir = r'C:\\Users\\ZX70WN\\Downloads'
features_file = os.path.join(download_dir, "column_name_list.txt")
model_dir = r'D:\\xgb_ALL_ric_ALL_horizon'
output_dir = r'D:\\xgb_softmax'

os.makedirs(output_dir, exist_ok=True)

for horizon in horizons:
    model_path = os.path.join(model_dir, f"model_one_horizon_{horizon}.model")
    
    # Assume function defined elsewhere:
    # main_df = get_merged_features(horizon, download_dir, features_file)
    main_df = pd.read_csv("dummy.csv")  # Replace this with actual merging logic

    y_col = f'utilization_t{horizon}'
    if y_col not in main_df.columns:
        continue

    X = main_df.drop(columns=[y_col, 'date', 'ric'])
    y = main_df[y_col]

    # Date splitting
    unique_dates = main_df['date'].sort_values().unique()
    n_dates = len(unique_dates)
    train_end = int(0.6 * n_dates)
    val_end = int(0.8 * n_dates)
    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:]

    train_idx = main_df['date'].isin(train_dates)
    val_idx = main_df['date'].isin(val_dates)
    test_idx = main_df['date'].isin(test_dates)

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    # Remove extremes (0 or 1)
    train_keep_idx = ~y_train.isin([0, 1])
    X_train = X_train[train_keep_idx]
    y_train = y_train[train_keep_idx]

    # Normalize
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_val_normalized = scaler.transform(X_val)
    X_test_normalized = scaler.transform(X_test)

    # Reconstruct DataFrames
    X_train = pd.DataFrame(X_train_normalized, columns=X.columns, index=X_train.index)
    X_val = pd.DataFrame(X_val_normalized, columns=X.columns, index=X_val.index)
    X_test = pd.DataFrame(X_test_normalized, columns=X.columns, index=X_test.index)

    # Bin target variable into 5 classes
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    y_binned = pd.cut(y, bins=bins, labels=False, include_lowest=True)
    num_classes = len(bins) - 1

    model = XGBClassifier(**params, num_class=num_classes)
    model.fit(X_train, y_binned[train_idx], 
              eval_set=[(X_val, y_binned[val_idx])], 
              early_stopping_rounds=10, verbose=True)

    # Save predictions
    y_pred = model.predict(X_val)

    pred_df = pd.DataFrame({
        'ric': main_df.loc[val_idx, 'ric'].values,
        'date': main_df.loc[val_idx, 'date'].values,
        'y_true': y_binned[val_idx].values,
        'y_pred': y_pred
    })
    pred_df.to_csv(os.path.join(output_dir, f"model_softmax_predicted_{horizon}.csv"), index=False)

    # Calculate and save metrics
    acc = accuracy_score(y_binned[val_idx], y_pred)
    metrics = {
        'accuracy': acc,
        'precision_micro': precision_score(y_binned[val_idx], y_pred, average='micro'),
        'precision_macro': precision_score(y_binned[val_idx], y_pred, average='macro'),
        'precision_weighted': precision_score(y_binned[val_idx], y_pred, average='weighted'),
        'recall_micro': recall_score(y_binned[val_idx], y_pred, average='micro'),
        'recall_macro': recall_score(y_binned[val_idx], y_pred, average='macro'),
        'recall_weighted': recall_score(y_binned[val_idx], y_pred, average='weighted'),
        'f1_micro': f1_score(y_binned[val_idx], y_pred, average='micro'),
        'f1_macro': f1_score(y_binned[val_idx], y_pred, average='macro'),
        'f1_weighted': f1_score(y_binned[val_idx], y_pred, average='weighted')
    }
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(os.path.join(output_dir, f"model_two_performance_metric_{horizon}.csv"), index=False)
"""

# Save to file so user can download
output_path = "/mnt/data/xgb_multiclass_with_metrics.py"
with open(output_path, "w") as f:
    f.write(code_string)

output_path
