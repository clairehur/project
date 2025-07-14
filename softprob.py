# ===== SETUP =====
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import os

# Define forecast horizons
horizons = [1, 5, 10]

# Set model directory
model_dir = r"D:\\xgb_ALL_ric_ALL_horizon"
output_dir = os.path.join(model_dir, "metrics_output")
os.makedirs(output_dir, exist_ok=True)

# Define XGBoost parameters (softprob)
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
    'objective': 'multi:softprob'
}

# ===== TRAINING & EVALUATION LOOP =====
for horizon in horizons:
    # Assume these are prepared by you
    # X_train, X_val, X_test
    # y_train, y_val, y_test

    # --- BINNING ---
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_centers = np.array([(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)])  # [0.1, 0.3, 0.5, 0.7, 0.9]

    # Binned targets
    y_train_binned = pd.cut(y_train, bins=bins, labels=False, include_lowest=True)
    y_val_binned = pd.cut(y_val, bins=bins, labels=False, include_lowest=True)
    y_test_binned = pd.cut(y_test, bins=bins, labels=False, include_lowest=True)

    # === MODEL: SOFTPROB ===
    model = XGBClassifier(**params, objective="multi:softprob", num_class=len(bin_centers))
    model.fit(X_train, y_train_binned, 
            eval_set=[(X_val, y_val_binned)], 
            early_stopping_rounds=10, verbose=True)

    # === PREDICT PROBABILITIES ===
    probs_val = model.predict_proba(X_val)
    probs_test = model.predict_proba(X_test)

    # === EXPECTED VALUE AS CONTINUOUS ESTIMATE ===
    y_val_pred_continuous = np.dot(probs_val, bin_centers)
    y_test_pred_continuous = np.dot(probs_test, bin_centers)

    # === REGRESSION METRICS ===
    val_rmse = mean_squared_error(y_val, y_val_pred_continuous, squared=False)
    val_mae = mean_absolute_error(y_val, y_val_pred_continuous)

    test_rmse = mean_squared_error(y_test, y_test_pred_continuous, squared=False)
    test_mae = mean_absolute_error(y_test, y_test_pred_continuous)

    # === SAVE RESULTS ===
    output_dir = os.path.join(model_dir, "regression_metrics_output")
    os.makedirs(output_dir, exist_ok=True)

    metrics = pd.DataFrame([{
        "horizon": horizon,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "test_rmse": test_rmse,
        "test_mae": test_mae
    }])
    metrics.to_csv(os.path.join(output_dir, f"regression_metrics_horizon_{horizon}.csv"), index=False)
