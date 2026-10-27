"""
Logistic regression for binary classification on the (offline) Wisconsin breast cancer dataset.

This script demonstrates:
    1. A leakage-free scikit-learn Pipeline (StandardScaler -> LogisticRegression).
    2. Choosing the regularisation strength C and the L1/L2 mix (l1_ratio) with stratified cross-validation.
    3. Evaluation beyond accuracy: confusion matrix, precision/recall/F1, ROC-AUC and average precision.
    4. Probability calibration: reliability diagram and Brier score, compared with a badly calibrated
       Gaussian naive Bayes model and its isotonic-calibrated version.
    5. Decision threshold selection: the default threshold 0.5 is rarely optimal. We pick a threshold on
       out-of-fold training predictions (never on the test set) that achieves a target recall.

The positive class (1) is "malignant", since missing a malignant tumour (false negative) is the costly error.
Plots are saved to the outputs/ folder next to this script.

How to run (from the repository root):
    python -m p05_ml_classic.c00_linear_models.main_2_logistic_regression
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SEED = 42
TARGET_RECALL = 0.98  # business requirement: catch at least 98% of malignant tumours


def build_pipeline() -> Pipeline:
    """Scaling + logistic regression. The saga solver supports both L1 and L2 (elastic-net) penalties."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="saga", max_iter=10_000, random_state=SEED)),
        ]
    )


def threshold_for_recall(y_true: np.ndarray, proba: np.ndarray, target_recall: float) -> float:
    """
    Return the largest probability threshold whose recall is still >= target_recall.

    A larger threshold means fewer positive predictions (higher precision), so we want the largest threshold
    that still meets the recall requirement.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision/recall have one more element than thresholds; drop the last point (recall=0)
    ok = recall[:-1] >= target_recall
    return float(thresholds[ok].max()) if ok.any() else 0.0


def threshold_max_f1(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Return the probability threshold that maximises the F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[np.argmax(f1)])


def report_at_threshold(name: str, y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    """Confusion-matrix based metrics for a given threshold."""
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "rule": name,
        "threshold": round(threshold, 3),
        "TP": tp,
        "FN (missed cancers)": fn,
        "FP (false alarms)": fp,
        "TN": tn,
        "precision": round(tp / max(tp + fp, 1), 3),
        "recall": round(tp / max(tp + fn, 1), 3),
    }


def plot_roc_pr(y_test: np.ndarray, probas: dict[str, np.ndarray], path: str) -> None:
    """ROC and precision-recall curves for several models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        prec, rec, _ = precision_recall_curve(y_test, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, p):.3f})")
        axes[1].plot(rec, prec, label=f"{name} (AP={average_precision_score(y_test, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="random")
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate (recall)", title="ROC curve")
    axes[1].axhline(y_test.mean(), color="k", ls="--", lw=0.8, label="random (prevalence)")
    axes[1].set(xlabel="recall", ylabel="precision", title="Precision-recall curve")
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_calibration(y_test: np.ndarray, probas: dict[str, np.ndarray], path: str) -> None:
    """Reliability diagram: predicted probability vs observed frequency of the positive class."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="perfectly calibrated")
    for name, p in probas.items():
        frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=8, strategy="quantile")
        axes[0].plot(mean_pred, frac_pos, "o-", label=f"{name} (Brier={brier_score_loss(y_test, p):.3f})")
        axes[1].hist(p, bins=20, range=(0, 1), histtype="step", lw=2, label=name)
    axes[0].set(xlabel="mean predicted probability", ylabel="fraction of positives", title="Reliability diagram")
    axes[1].set(xlabel="predicted probability", ylabel="count", title="Distribution of predicted probabilities")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_threshold_curve(y_true: np.ndarray, proba: np.ndarray, chosen: dict[str, float], path: str) -> None:
    """Precision and recall as a function of the decision threshold, with the chosen thresholds marked."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, precision[:-1], label="precision")
    ax.plot(thresholds, recall[:-1], label="recall")
    for (name, t), c in zip(chosen.items(), ["grey", "green", "purple"]):
        ax.axvline(t, color=c, ls="--", lw=1, label=f"{name}: {t:.2f}")
    ax.set(xlabel="decision threshold", ylabel="score", title="Precision/recall vs threshold (out-of-fold, train)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data; relabel so that 1 = malignant (the original dataset uses 1 = benign)
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = (data.target == 0).astype(int).to_numpy()
    print(f"Breast cancer dataset: {X.shape[0]} samples, {X.shape[1]} features, {y.mean():.1%} malignant")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

    # 2. Hyperparameter search over C (inverse regularisation strength) and the L1/L2 mix.
    #    l1_ratio=0 -> pure L2 (ridge-like), l1_ratio=1 -> pure L1 (lasso-like, sparse coefficients).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    param_grid = {"clf__C": np.logspace(-3, 2, 11), "clf__l1_ratio": [0.0, 0.5, 1.0]}
    search = GridSearchCV(build_pipeline(), param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    n_nonzero = int(np.sum(np.abs(best["clf"].coef_) > 1e-6))
    print(f"\nBest params (CV ROC-AUC={search.best_score_:.4f}): {search.best_params_}")
    print(f"Non-zero coefficients: {n_nonzero}/{X.shape[1]}")
    coefs = pd.Series(best["clf"].coef_.ravel(), index=X.columns)
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(8)
    print("Largest standardised coefficients (log-odds change per 1 std of the feature):")
    print(top.round(3).to_string())

    # 3. Test-set evaluation at the default threshold of 0.5
    proba_lr = best.predict_proba(X_test)[:, 1]
    print("\nClassification report at threshold 0.5 (test set):")
    print(classification_report(y_test, (proba_lr >= 0.5).astype(int), target_names=["benign", "malignant"], digits=3))
    print(f"ROC-AUC={roc_auc_score(y_test, proba_lr):.4f}, average precision={average_precision_score(y_test, proba_lr):.4f}")

    # 4. Calibration: logistic regression vs Gaussian naive Bayes (raw and isotonic-calibrated)
    nb = GaussianNB().fit(X_train, y_train)
    nb_cal = CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5).fit(X_train, y_train)
    probas = {
        "logistic regression": proba_lr,
        "Gaussian NB": nb.predict_proba(X_test)[:, 1],
        "Gaussian NB + isotonic": nb_cal.predict_proba(X_test)[:, 1],
    }
    print("\nBrier score (lower is better; measures calibration + discrimination):")
    for name, p in probas.items():
        print(f"  {name:24s} Brier={brier_score_loss(y_test, p):.4f}  ROC-AUC={roc_auc_score(y_test, p):.4f}")
    plot_roc_pr(y_test, probas, os.path.join(OUTPUT_DIR, "logreg_roc_pr.png"))
    plot_calibration(y_test, probas, os.path.join(OUTPUT_DIR, "logreg_calibration.png"))

    # 5. Threshold selection on OUT-OF-FOLD predictions of the training set (the test set stays untouched)
    oof_proba = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    thresholds = {
        "default": 0.5,
        "max F1": threshold_max_f1(y_train, oof_proba),
        f"recall>={TARGET_RECALL}": threshold_for_recall(y_train, oof_proba, TARGET_RECALL),
    }
    plot_threshold_curve(y_train, oof_proba, thresholds, os.path.join(OUTPUT_DIR, "logreg_threshold.png"))
    rows = [report_at_threshold(name, y_test, proba_lr, t) for name, t in thresholds.items()]
    print("\nEffect of the decision threshold on the test set:")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nPlots saved to {OUTPUT_DIR}")
