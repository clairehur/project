from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Example horizons
horizons = [1, 5, 10]  # <- define your horizons here

# Parameters
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

model_dir = r"D:\\xgb_ALL_ric_ALL_horizon"
output_dir = os.path.join(model_dir, "metrics_output")
os.makedirs(output_dir, exist_ok=True)

for horizon in horizons:

    # === Assume you already have your X_train, y_train, etc. loaded ===
    # Replace these with your actual preprocessing pipeline before this point

    # Example placeholders (REMOVE THESE in your real pipeline):
    # X_train, X_val, X_test = ...
    # y_train, y_val, y_test = ...

    # Binning
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    y_train_binned = pd.cut(y_train, bins=bins, labels=False, include_lowest=True)
    y_val_binned = pd.cut(y_val, bins=bins, labels=False, include_lowest=True)
    y_test_binned = pd.cut(y_test, bins=bins, labels=False, include_lowest=True)

    # Train model
    model = XGBClassifier(**params, num_class=5)
    model.fit(X_train, y_train_binned, 
              eval_set=[(X_val, y_val_binned)], 
              early_stopping_rounds=10, verbose=True)

    # Predict
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Save predictions for confusion matrix plotting later
    pd.DataFrame({"true": y_val_binned, "pred": y_val_pred}).to_csv(
        os.path.join(output_dir, f"val_preds_horizon_{horizon}.csv"), index=False)
    pd.DataFrame({"true": y_test_binned, "pred": y_test_pred}).to_csv(
        os.path.join(output_dir, f"test_preds_horizon_{horizon}.csv"), index=False)

    # Save performance metrics
    def save_metrics(y_true, y_pred, split, output_dir, horizon):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_micro': precision_score(y_true, y_pred, average='micro'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_micro': recall_score(y_true, y_pred, average='micro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        }

        report = classification_report(y_true, y_pred, output_dict=True)
        f1_per_class = {f"f1_class_{k}": v["f1-score"] for k, v in report.items() if k.isdigit()}
        metrics.update(f1_per_class)

        df_metrics = pd.DataFrame([metrics])
        df_metrics.to_csv(os.path.join(output_dir, f"model_metrics_{split}_horizon_{horizon}.csv"), index=False)

    # Save for val and test
    save_metrics(y_val_binned, y_val_pred, "val", output_dir, horizon)
    save_metrics(y_test_binned, y_test_pred, "test", output_dir, horizon)


# ===== Metric Plotting =====
import matplotlib.pyplot as plt
import seaborn as sns

# Collect metrics
metric_dfs = []

for horizon in horizons:
    for split in ['val', 'test']:
        path = os.path.join(output_dir, f"model_metrics_{split}_horizon_{horizon}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['horizon'] = horizon
            df['split'] = split
            metric_dfs.append(df)

all_metrics_df = pd.concat(metric_dfs, ignore_index=True)

# Melt for visualization
melted = all_metrics_df.melt(
    id_vars=["horizon", "split"], 
    value_vars=[
        "accuracy", 
        "precision_micro", "recall_micro", "f1_micro", 
        "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted"
    ],
    var_name="metric", value_name="value"
)

# Plot
plt.figure(figsize=(16, 8))
sns.barplot(data=melted, x="horizon", y="value", hue="metric")
plt.title("Performance Metrics per Horizon (Val + Test)")
plt.ylabel("Score")
plt.xlabel("Forecast Horizon")
plt.xticks(rotation=0)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ===== Confusion Matrix for test data set =====

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

for horizon in horizons:
    pred_path = os.path.join(output_dir, f"test_preds_horizon_{horizon}.csv")
    if not os.path.exists(pred_path):
        print(f"{pred_path} not found, skipping...")
        continue

    df_pred = pd.read_csv(pred_path)
    y_true = df_pred["true"]
    y_pred = df_pred["pred"]

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    plt.figure(figsize=(6, 5))
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f"Confusion Matrix — Test (Horizon {horizon})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()
