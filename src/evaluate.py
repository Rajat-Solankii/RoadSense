"""
RoadSense AI: Evaluation Metrics Module
Standardized metric computation for:
  - Task A: Binary classification (Macro-F1, Dangerous Recall/Precision, ROC-AUC, PR-AUC)
  - Task B: 3-Class classification (Macro-F1, Fatal Recall/Precision, OVR ROC-AUC, PR-AUC)
  - Task C: Regression (MAE, Median AE, RMSE, R2, Poisson deviance)
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    r2_score,
    d2_tweedie_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize


class PlattCalibrator:
    """
    Sigmoid-based Platt scaling fitted on validation probabilities.
    Recalibrates distorted probabilities caused by class weighting or resampling.
    """
    def __init__(self):
        self.calibrator = LogisticRegression(C=1.0, solver="lbfgs")
        self.is_fitted = False
        
    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        eps = 1e-6
        p = np.clip(probs, eps, 1.0 - eps)
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        self.calibrator.fit(logits, y_true)
        self.is_fitted = True
        return self
        
    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return probs
        eps = 1e-6
        p = np.clip(probs, eps, 1.0 - eps)
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        return self.calibrator.predict_proba(logits)[:, 1]


def evaluate_binary_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes all standard metrics for Task A (Binary: 1=Dangerous, 0=Minor).
    """
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "dangerous_f1": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "dangerous_precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "dangerous_recall": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
    }
    
    if y_prob is not None:
        # If 2D probability array passed, extract positive class probability
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            prob_pos = y_prob[:, 1]
        else:
            prob_pos = y_prob
            
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, prob_pos)), 4)
        except Exception:
            metrics["roc_auc"] = 0.5
            
        try:
            metrics["pr_auc"] = round(float(average_precision_score(y_true, prob_pos)), 4)
        except Exception:
            metrics["pr_auc"] = round(float(np.mean(y_true)), 4)
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


def evaluate_multiclass_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    classes: List[int] = [0, 1, 2] # 0: Fatal, 1: Serious, 2: Slight
) -> Dict[str, Any]:
    """
    Computes all standard metrics for Task B (3-class: 0=Fatal, 1=Serious, 2=Slight).
    """
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "fatal_precision": round(float(precision_score(y_true, y_pred, labels=[0], average=None, zero_division=0)[0]), 4),
        "fatal_recall": round(float(recall_score(y_true, y_pred, labels=[0], average=None, zero_division=0)[0]), 4),
        "fatal_f1": round(float(f1_score(y_true, y_pred, labels=[0], average=None, zero_division=0)[0]), 4),
        "serious_precision": round(float(precision_score(y_true, y_pred, labels=[1], average=None, zero_division=0)[0]), 4),
        "serious_recall": round(float(recall_score(y_true, y_pred, labels=[1], average=None, zero_division=0)[0]), 4),
        "serious_f1": round(float(f1_score(y_true, y_pred, labels=[1], average=None, zero_division=0)[0]), 4),
        "slight_precision": round(float(precision_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]), 4),
        "slight_recall": round(float(recall_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]), 4),
        "slight_f1": round(float(f1_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]), 4),
    }
    
    if y_prob is not None:
        try:
            metrics["roc_auc_ovr"] = round(float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")), 4)
        except Exception:
            metrics["roc_auc_ovr"] = 0.5
            
        try:
            y_bin = label_binarize(y_true, classes=classes)
            metrics["pr_auc_macro"] = round(float(average_precision_score(y_bin, y_prob, average="macro")), 4)
            # Fatal class specific PR-AUC (class index 0)
            metrics["fatal_pr_auc"] = round(float(average_precision_score(y_bin[:, 0], y_prob[:, 0])), 4)
        except Exception:
            metrics["pr_auc_macro"] = None
            metrics["fatal_pr_auc"] = None
    else:
        metrics["roc_auc_ovr"] = None
        metrics["pr_auc_macro"] = None
        metrics["fatal_pr_auc"] = None
        
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=classes).tolist()
    return metrics


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Computes all standard metrics for Task C (Casualties count).
    Includes MAE, Median AE (robust to extreme outliers), RMSE, and R2.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    med_ae = float(median_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    
    return {
        "mae": round(mae, 4),
        "median_ae": round(med_ae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4)
    }
