import numpy as np

def emd_loss(preds, dtrain):
    """
    Custom ordinal-aware loss function based on cumulative probability distance.
    """
    labels = dtrain.get_label().astype(int)
    n_classes = 5
    preds = preds.reshape(-1, n_classes)

    # Softmax to get predicted probabilities
    exps = np.exp(preds - np.max(preds, axis=1, keepdims=True))
    probs = exps / np.sum(exps, axis=1, keepdims=True)

    # Create cumulative distributions
    cdf_preds = np.cumsum(probs, axis=1)
    cdf_true = np.zeros_like(cdf_preds)
    cdf_true[np.arange(len(labels)), labels] = 1
    cdf_true = np.cumsum(cdf_true, axis=1)

    # Gradient and Hessian
    grad = cdf_preds - cdf_true
    hess = probs * (1 - probs)  # diagonal approximation

    return grad.ravel(), hess.ravel()

import xgboost as xgb
from sklearn.metrics import f1_score, classification_report

def train_xgb_with_emd(X_train, y_train, X_val, y_val, num_class=5):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        'max_depth': 6,
        'eta': 0.1,
        'num_class': num_class,
        'objective': 'multi:softprob',  # needed for shape, overridden by custom obj
        'eval_metric': 'mlogloss'       # just for logging
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, 'validation')],
        obj=emd_loss,
        early_stopping_rounds=20,
        verbose_eval=10
    )

    return model

dtest = xgb.DMatrix(X_test)
probs = model.predict(dtest)
preds = np.argmax(probs, axis=1)

# Evaluate
print(classification_report(y_test, preds))
