"""ML-based channel classifier using SIC code features."""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.preprocessing import MultiLabelBinarizer


# ------------------------------------------------------------------
# Feature preparation
# ------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, MultiLabelBinarizer]:
    """Filter to active companies and one-hot encode SIC codes.

    Returns
    -------
    X : pd.DataFrame
        One-hot encoded SIC code features.
    y : pd.Series
        Channel labels.
    mlb : MultiLabelBinarizer
        Fitted binarizer (needed for inference on new data).
    """
    # Filter to active companies only
    active = df[df["company_status"] == "active"].copy()
    print(f"  Active companies: {len(active)} / {len(df)} total")

    if active.empty:
        print("  [WARN] No active companies found — using all companies instead.")
        active = df.copy()

    # Parse SIC codes
    active["sic_list"] = active["sic_codes"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
    )

    # Drop companies with no SIC codes
    active = active[active["sic_list"].apply(len) > 0]
    print(f"  Companies with SIC codes: {len(active)}")

    if active.empty:
        raise ValueError("No companies with SIC codes to train on.")

    mlb = MultiLabelBinarizer()
    X = pd.DataFrame(
        mlb.fit_transform(active["sic_list"]),
        columns=mlb.classes_,
        index=active.index,
    )
    y = active["channel"]

    return X, y, mlb


# ------------------------------------------------------------------
# Training and evaluation
# ------------------------------------------------------------------

def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    n_splits: int = 5,
) -> dict:
    """Train a Random Forest classifier with stratified cross-validation.

    Saves model, confusion matrix plot, and text report to *output_dir*.
    """
    labels = sorted(y.unique())

    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # --- Cross-validation ---
    cv = StratifiedKFold(n_splits=min(n_splits, y.value_counts().min(), len(y)),
                         shuffle=True, random_state=42)

    scoring = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    cv_results = cross_validate(clf, X, y, cv=cv, scoring=scoring, return_train_score=False)

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("ML MODEL EVALUATION (Random Forest)")
    lines.append("=" * 80)
    lines.append(f"\nDataset: {len(y)} active companies, {X.shape[1]} SIC features")
    lines.append(f"Classes: {labels}")
    lines.append(f"Class distribution:\n{y.value_counts().to_string()}")
    lines.append(f"\nStratified {cv.get_n_splits()}-Fold Cross-Validation:")
    for metric in scoring:
        vals = cv_results[f"test_{metric}"]
        lines.append(f"  {metric:>20s}: {vals.mean():.3f} ± {vals.std():.3f}")

    # --- Train on full data for final model + confusion matrix ---
    clf.fit(X, y)
    y_pred = clf.predict(X)  # train set predictions (for illustrative confusion matrix)

    report = classification_report(y, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=labels)

    lines.append(f"\nFull-data classification report (train set):\n{report}")
    lines.append(f"Confusion matrix (train set):\n{cm}")

    # Feature importance — top SIC codes
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(15)
    lines.append("\nTop 15 most important SIC codes:")
    for sic, imp in top_features.items():
        lines.append(f"  {sic}: {imp:.4f}")

    result_text = "\n".join(lines)
    print(result_text)

    # --- Save outputs ---
    report_path = output_dir / "ml_model_results.txt"
    report_path.write_text(result_text)
    print(f"\n  Saved ML report → {report_path}")

    # Confusion matrix heatmap
    _plot_confusion_matrix(cm, labels, output_dir / "confusion_matrix.png")

    # Feature importance chart
    _plot_feature_importance(top_features, output_dir / "feature_importance.png")

    # Save model
    model_path = output_dir / "model.pkl"
    joblib.dump({"model": clf, "binarizer": mlb, "labels": labels}, model_path)
    print(f"  Saved model → {model_path}")

    return {
        "cv_results": cv_results,
        "accuracy": accuracy_score(y, y_pred),
        "classification_report": report,
        "confusion_matrix": cm,
        "model": clf,
    }


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

def _plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — ML Model")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix → {output_path}")


def _plot_feature_importance(top_features: pd.Series, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    top_features.sort_values().plot.barh(ax=ax, color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title("Top SIC Code Feature Importances")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved feature importance chart → {output_path}")
