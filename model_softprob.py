import os
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set up global paths and constants
output_dir = 'D:/xgb_softprob_ALL_ric_ALL_horizon'
os.makedirs(output_dir, exist_ok=True)

# Define the Optuna objective function
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 0.01, 1.0),
        "alpha": trial.suggest_float("alpha", 0.01, 1.0),
        "use_label_encoder": False,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss"
    }

    model = xgb.XGBClassifier(**params, num_class=len(bin_centers))

    model.fit(
        X_train, y_train_binned,
        eval_set=[(X_train, y_train_binned), (X_val, y_val_binned)],
        early_stopping_rounds=10,
        verbose=False
    )

    evals_result = model.evals_result()

    probs_val = model.predict_proba(X_val)
    y_val_pred_continuous = np.dot(probs_val, bin_centers)
    val_rmse = mean_squared_error(y_val, y_val_pred_continuous, squared=False)

    # Save eval logs
    logs_df = pd.DataFrame({
        'iteration': range(len(evals_result['validation_0']['mlogloss'])),
        'train_logloss': evals_result['validation_0']['mlogloss'],
        'val_logloss': evals_result['validation_1']['mlogloss']
    })
    csv_filename = f'{output_dir}/logs_trial_{trial.number}.csv'
    logs_df.to_csv(csv_filename, index=False)

    return val_rmse

# Run Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)

# Save all trial results
results = []
for t in study.trials:
    row = t.params.copy()
    row['value'] = t.value
    results.append(row)

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, 'optuna_trials_summary.csv'), index=False)

# Save best trial
best_params = study.best_trial.params.copy()
best_params['value'] = study.best_trial.value
pd.DataFrame([best_params]).to_csv(
    os.path.join(output_dir, 'optuna_best_trial.csv'), index=False)

# Train final model on best params
final_model = xgb.XGBClassifier(**study.best_trial.params, num_class=len(bin_centers), eval_metric="mlogloss")
final_model.fit(
    X_train, y_train_binned,
    eval_set=[(X_val, y_val_binned)],
    early_stopping_rounds=10,
    verbose=True
)

# Predict on validation and test
probs_val = final_model.predict_proba(X_val)
probs_test = final_model.predict_proba(X_test)

y_val_pred_continuous = np.dot(probs_val, bin_centers)
y_test_pred_continuous = np.dot(probs_test, bin_centers)

val_rmse = mean_squared_error(y_val, y_val_pred_continuous, squared=False)
val_mae = mean_absolute_error(y_val, y_val_pred_continuous)
test_rmse = mean_squared_error(y_test, y_test_pred_continuous, squared=False)
test_mae = mean_absolute_error(y_test, y_test_pred_continuous)

metrics = pd.DataFrame([{
    'horizon': horizon,
    'val_rmse': val_rmse,
    'val_mae': val_mae,
    'test_rmse': test_rmse,
    'test_mae': test_mae
}])

metrics.to_csv(os.path.join(output_dir, f'regression_metrics_horizon_{horizon}.csv'), index=False)
print(metrics)