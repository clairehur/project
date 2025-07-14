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

# Save performance metrics
def save_metrics(y_true, y_pred, split, output_dir, horizon):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    import pandas as pd
    import os

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
    os.makedirs(output_dir, exist_ok=True)
    df_metrics.to_csv(os.path.join(output_dir, f"model_metrics_{split}_horizon_{horizon}.csv"), index=False)

# Output directory
output_dir = os.path.join(model_dir, "metrics_output")

# Save for val and test
save_metrics(y_val_binned, y_val_pred, "val", output_dir, horizon)
save_metrics(y_test_binned, y_test_pred, "test", output_dir, horizon)
