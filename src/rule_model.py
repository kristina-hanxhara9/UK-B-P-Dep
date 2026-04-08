"""Rule-based channel classifier using SIC code weights and data-driven name keywords."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.analysis import compute_channel_sic_sets
from src.name_features import keyword_channel_scores


# ------------------------------------------------------------------
# Rule construction
# ------------------------------------------------------------------

def build_rules(
    matrix: pd.DataFrame,
    keyword_data: dict | None = None,
) -> dict:
    """Derive classification rules from the SIC x channel matrix.

    Rules
    -----
    For each SIC code, compute a CLASS-SIZE-NORMALISED weight per channel:

        raw_rate(sic, ch) = count(sic, ch) / total_companies_in_ch
        weight(sic, ch) = raw_rate(sic, ch) / sum(raw_rate(sic, *))

    This means: "what fraction of builders merchants have this SIC code?"
    vs "what fraction of plumbing merchants have it?" -- regardless of
    how many companies are in each input sheet.

    Without this normalisation, a channel with 800 companies always
    dominates over one with 50.
    """
    numeric = matrix.drop(columns=["description"], errors="ignore")

    # Total companies per channel (column sums)
    channel_totals = numeric.sum(axis=0)
    # Avoid division by zero
    channel_totals = channel_totals.replace(0, 1)

    sic_weights: dict[str, dict[str, float]] = {}
    for sic in numeric.index:
        # Rate: what proportion of each channel's companies have this SIC
        rates = {}
        for channel in numeric.columns:
            rates[channel] = float(numeric.loc[sic, channel]) / float(channel_totals[channel])

        rate_sum = sum(rates.values())
        if rate_sum == 0:
            continue

        # Normalise rates to sum to 1 (so they're comparable)
        sic_weights[sic] = {
            ch: rate / rate_sum for ch, rate in rates.items()
        }

    # Fallback: channel with the most companies overall
    fallback = numeric.sum(axis=0).idxmax()

    rules = {"sic_weights": sic_weights, "fallback": fallback}

    if keyword_data and keyword_data.get("keyword_scores"):
        rules["keyword_scores"] = keyword_data["keyword_scores"]

    return rules


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------

def predict_channel(
    sic_codes: list[str],
    rules: dict,
    company_name: str = "",
    name_weight: float = 0.3,
) -> str:
    """Predict a single company's channel from SIC codes + name keywords.

    The final score blends SIC-based weights and name-keyword scores:
        score = (1 - name_weight) * sic_score + name_weight * keyword_score

    Name keywords are only used if ``keyword_scores`` were discovered from
    the data and stored in the rules.  Otherwise, SIC-only.
    """
    weights = rules["sic_weights"]
    sic_scores: dict[str, float] = {}

    for sic in sic_codes:
        if sic in weights:
            for ch, w in weights[sic].items():
                sic_scores[ch] = sic_scores.get(ch, 0.0) + w

    # Name keyword scores - only if discovered from data
    kw_scores: dict[str, float] = {}
    if company_name and "keyword_scores" in rules:
        kw_scores = keyword_channel_scores(company_name, rules["keyword_scores"])

    # Blend
    all_channels = set(list(sic_scores.keys()) + list(kw_scores.keys()))
    if not all_channels:
        return rules["fallback"]

    # If we have no keyword data, use SIC only
    use_name = bool(kw_scores)
    combined: dict[str, float] = {}
    for ch in all_channels:
        s = sic_scores.get(ch, 0.0)
        if use_name:
            k = kw_scores.get(ch, 0.0)
            combined[ch] = (1 - name_weight) * s + name_weight * k
        else:
            combined[ch] = s

    return max(combined, key=combined.get)


def predict_all(df: pd.DataFrame, rules: dict) -> pd.Series:
    """Predict channels for all companies in the DataFrame."""
    def _predict_row(row):
        sic = json.loads(row["sic_codes"]) if isinstance(row["sic_codes"], str) else row["sic_codes"]
        name = str(row.get("matched_name") or row.get("input_name") or "")
        return predict_channel(sic, rules, company_name=name)

    return df.apply(_predict_row, axis=1)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_rule_model(
    df: pd.DataFrame,
    rules: dict,
) -> dict:
    """Evaluate the rule-based model against ground-truth channel labels."""
    predictions = predict_all(df, rules)
    y_true = df["channel"]
    y_pred = predictions

    labels = sorted(y_true.unique())
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    acc = float((y_true == y_pred).mean())
    print(f"  Rule model accuracy: {acc:.2%}")
    print(f"\n{report}")

    return {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "predictions": predictions,
    }
