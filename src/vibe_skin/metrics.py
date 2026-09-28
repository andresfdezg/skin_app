"""Classification metrics for skin lesions."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

MALIGNANT = ["akiec", "bcc", "mel"]


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, classes: list[str]) -> dict:
    y_pred = probs.argmax(1)
    labels = list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {}
    for i, c in enumerate(classes):
        is_c = y_true == i
        tn = cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]
        per_class[c] = {
            "support": int(support[i]),
            "sensitivity": float(recall[i]),
            "specificity": float(tn / (~is_c).sum()),
            "precision": float(precision[i]),
            "f1": float(f1[i]),
            "auc": float(roc_auc_score(is_c, probs[:, i])) if 0 < is_c.sum() < len(is_c) else None,
        }

    # Malignant (mel + bcc + akiec) vs benign: the question a triage app really answers
    mal_idx = [classes.index(c) for c in MALIGNANT]
    y_mal = np.isin(y_true, mal_idx)
    p_mal = probs[:, mal_idx].sum(1)

    aucs = [v["auc"] for v in per_class.values() if v["auc"] is not None]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_auc": float(np.mean(aucs)),
        "melanoma_sensitivity": per_class["mel"]["sensitivity"],
        "melanoma_specificity": per_class["mel"]["specificity"],
        "melanoma_auc": per_class["mel"]["auc"],
        "malignant_auc": float(roc_auc_score(y_mal, p_mal)),
        "malignant_sensitivity": float((np.isin(y_pred, mal_idx) & y_mal).sum() / y_mal.sum()),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }


def plot_confusion(cm: np.ndarray, classes: list[str], path: Path, title: str = "") -> None:
    cm = np.asarray(cm)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion matrix (row-normalized)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                    fontsize=8, color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summary_line(m: dict) -> str:
    return (f"bal_acc {m['balanced_accuracy']:.3f} | macro_auc {m['macro_auc']:.3f} | "
            f"mel_sens {m['melanoma_sensitivity']:.3f} | mal_auc {m['malignant_auc']:.3f}")
