from typing import List, Tuple, Dict, Any, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (y_score >= threshold).astype(np.int32)
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / max(1, (fp + tn))
    tpr = tp / max(1, (tp + fn))

    return {
        "conf_matrix": cm.tolist(),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "fpr": float(fpr),
        "tpr": float(tpr),
    }

def select_threshold_youden(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    fpr, tpr, thr = roc_curve(y_true, y_score, drop_intermediate=False)
    j = tpr - fpr
    i = int(np.argmax(j))
    return {
        "threshold": float(thr[i]),
        "fpr": float(fpr[i]),
        "tpr": float(tpr[i]),
        "youden_j": float(j[i]),
    }

def select_threshold_np_quantile(y_true: np.ndarray, y_score: np.ndarray, alpha: float) -> Dict[str, float]:
    neg = y_score[y_true == 0]
    tau = float(np.quantile(neg, 1.0 - alpha, method="higher"))
    y_pred = (y_score >= tau).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / max(1, fp + tn)
    tpr = tp / max(1, tp + fn)
    return {
        "threshold": tau,
        "fpr": float(fpr),
        "tpr": float(tpr),
        "n_neg": int(len(neg)),
    }

def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> Dict[str, float]:
    """
    Return interpolated TPR at a target FPR.
    The primary return value preserves the original interpolated statistic.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, drop_intermediate=False)
    tpr_interp = float(np.interp(target_fpr, fpr, tpr))
    return {"tpr_interp": tpr_interp}

def multi_tpr_at_fprs(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fprs: List[float],
) -> Dict[str, Dict[str, float]]:
    results = {}
    for alpha in target_fprs:
        key = f"{alpha:.6f}".rstrip("0").rstrip(".")
        results[key] = tpr_at_fpr(y_true, y_score, alpha)
    return results

def format_fpr_key(alpha: float) -> str:
    return f"{alpha:.6f}".rstrip("0").rstrip(".")
