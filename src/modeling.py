# src/modeling.py

from typing import Dict, Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier


def build_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Create the dictionary of models to evaluate.
    """
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            objective="multi:softprob",
        ),
    }
    return models


def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    """
    Train each model and print evaluation metrics.
    Returns:
        results : dict mapping model_name -> metrics + model object
    """
    results = {}

    for name, model in models.items():
        print("\n" + "=" * 60)
        print(f"[INFO] Training model: {name}")
        print("=" * 60)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Classification report (text)
        cr_text = classification_report(y_test, y_pred, digits=3)
        print(cr_text)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Multi-class AUC (One-vs-Rest), if predict_proba is available
        try:
            y_proba = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
        except Exception:
            auc = None

        if auc is not None:
            print(f"\n{name} ROC-AUC (OvR): {auc:.3f}")
        else:
            print(f"\n{name} ROC-AUC (OvR): N/A (no predict_proba)")

        results[name] = {
            "model": model,
            "classification_report_text": cr_text,
            "confusion_matrix": cm,
            "auc": auc,
        }

    return results
