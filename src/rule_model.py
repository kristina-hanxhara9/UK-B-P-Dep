"""Rule-based channel classifier using SIC code weights."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.analysis import unique_sic_codes, compute_channel_sic_sets


# ------------------------------------------------------------------
# Rule construction
# ------------------------------------------------------------------

def build_rules(matrix: pd.DataFrame) -> dict:
    """Derive classification rules from the SIC × channel matrix.

    Rules
    -----
    For each SIC code, compute a weight per channel:
        weight(sic, channel) = count(sic, channel) / total(sic)
    This gives the probability that a company with this SIC code belongs
    to each channel.

    Returns a dict:  {sic_code: {channel: weight, ...}, ...}
    plus a "fallback" key with the most common channel.
    """
    numeric = matrix.drop(columns=["description"], errors="ignore")
    row_totals = numeric.sum(axis=1)

    rules: dict[str, dict[str, float]] = {}
    for sic in numeric.index:
        total = row_totals[sic]
        if total == 0:
            continue
        rules[sic] = {
            channel: float(numeric.loc[sic, channel] / total)
            for channel in numeric.columns
        }

    # Fallback: channel with the most companies overall
    fallback = numeric.sum(axis=0).idxmax()

    return {"sic_weights": rules, "fallback": fallback}


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------

def predict_channel(sic_codes: list[str], rules: dict) -> str:
    """Predict a single company's channel from its SIC codes."""
    weights = rules["sic_weights"]
    channels = set()
    scores: dict[str, float] = {}

    for sic in sic_codes:
        if sic in weights:
            for ch, w in weights[sic].items():
                channels.add(ch)
                scores[ch] = scores.get(ch, 0.0) + w

    if not scores:
        return rules["fallback"]

    return max(scores, key=scores.get)


def predict_all(df: pd.DataFrame, rules: dict) -> pd.Series:
    """Predict channels for all companies in the DataFrame."""
    def _predict_row(row):
        sic = json.loads(row["sic_codes"]) if isinstance(row["sic_codes"], str) else row["sic_codes"]
        return predict_channel(sic, rules)

    return df.apply(_predict_row, axis=1)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_rule_model(
    df: pd.DataFrame,
    rules: dict,
    output_path: Path,
) -> dict:
    """Evaluate the rule-based model against ground-truth channel labels."""
    predictions = predict_all(df, rules)
    y_true = df["channel"]
    y_pred = predictions

    labels = sorted(y_true.unique())
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("RULE-BASED MODEL EVALUATION")
    lines.append("=" * 80)
    lines.append(f"\nAccuracy: {(y_true == y_pred).mean():.2%}")
    lines.append(f"\n{report}")
    lines.append("\nConfusion Matrix:")
    lines.append(f"Labels: {labels}")
    lines.append(str(cm))

    # Show unique SIC rules
    sic_sets = compute_channel_sic_sets(
        pd.DataFrame(rules["sic_weights"]).T.fillna(0)
    ) if rules["sic_weights"] else {}

    result_text = "\n".join(lines)
    output_path.write_text(result_text)
    print(result_text)
    print(f"\n  Saved rule model results → {output_path}")

    return {
        "accuracy": float((y_true == y_pred).mean()),
        "classification_report": report,
        "confusion_matrix": cm,
        "predictions": predictions,
    }
