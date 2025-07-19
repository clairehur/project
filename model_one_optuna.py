import seaborn as sns
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import optuna

# Objective function for Optuna
def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "tree_method": "auto",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.07),
        "max_depth": trial.suggest_int("max_depth", 4, 6),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 4),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.8),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.7, 0.8),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.7, 0.8),
        "lambda": trial.suggest_float("lambda", 0.8, 1.2),
        "alpha": trial.suggest_float("alpha", 0.8, 1.2),
        "gamma": trial.suggest_int("gamma", 0, 1),
        "eval_metric": "rmse"
    }

    print(f"Trial {trial.number} with params: {params}")

    dtrain = xgb.DMatrix(data=X_train_normalized, label=y_train, feature_names=X_train.columns)
    dval = xgb.DMatrix(data=X_val_normalized, label=y_val, feature_names=X_val.columns)

    evals = [(dtrain, 'train'), (dval, 'val')]
    evals_result = {}

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=10,
        verbose_eval=5
    )

    train_rmse = evals_result['train']['rmse']
    val_rmse = evals_result['val']['rmse']

    logs_df = pd.DataFrame({
        'iteration': range(1, len(train_rmse) + 1),
        'train_rmse': train_rmse,
        'val_rmse': val_rmse
    })

    output_dir = 'D:/xgb_ALL_ric_ALL_horizon'
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = f'{output_dir}/logs_trial_{trial.number}.csv'
    logs_df.to_csv(csv_filename, index=False)

    val_pred = model.predict(dval)
    val_rmse_final = np.sqrt(mean_squared_error(y_val, val_pred))

    return val_rmse_final

# Train/Val/Test Splits and Scaling assumed ready here...
# Assume X_train, X_val, X_test, y_train, y_val, y_test already defined
# Apply standard scaling
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_val_normalized = scaler.transform(X_val)
X_test_normalized = scaler.transform(X_test)

X_train_normalized = pd.DataFrame(X_train_normalized, columns=X_train.columns, index=X_train.index)
X_val_normalized = pd.DataFrame(X_val_normalized, columns=X_val.columns, index=X_val.index)
X_test_normalized = pd.DataFrame(X_test_normalized, columns=X_test.columns, index=X_test.index)

# Run Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)

print("Best Trial:")
trial = study.best_trial
print(f"Value: {trial.value}")
print(f"Params: {trial.params}")

# Save all trial results
results = []
for t in study.trials:
    row = t.params.copy()
    row['value'] = t.value
    results.append(row)
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join("D:/xgb_ALL_ric_ALL_horizon", f"optuna_trials_horizon_{horizon}.csv"), index=False)

# Save best trial separately
best_params = study.best_trial.params.copy()
best_params['value'] = study.best_trial.value
best_df = pd.DataFrame([best_params])
best_df.to_csv(os.path.join("D:/xgb_ALL_ric_ALL_horizon", f"optuna_best_horizon_{horizon}.csv"), index=False)

# Train final model using best params
final_model = xgb.train(
    best_params,
    dtrain=xgb.DMatrix(data=X_train_normalized, label=y_train, feature_names=X_train.columns),
    num_boost_round=1000,
    evals=[(xgb.DMatrix(data=X_val_normalized, label=y_val, feature_names=X_val.columns), 'validation')],
    verbose_eval=5,
    early_stopping_rounds=10
)

# Predict on test set
dtest = xgb.DMatrix(data=X_test_normalized, label=y_test)
y_pred = final_model.predict(dtest)