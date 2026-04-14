"""Rule-based channel classifier using independent channel profiles.

Each channel is analysed SEPARATELY:
  - What SIC codes appear in this channel's companies, and at what rate?
  - What keywords appear in this channel's company names?

Channels do NOT compete against each other during analysis.
At prediction time, a company is scored against each channel's profile
independently, and assigned to the best-matching channel.
"""

import json

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.name_features import keyword_channel_scores


# ------------------------------------------------------------------
# Build independent channel profiles
# ------------------------------------------------------------------

def build_rules(
    matrix: pd.DataFrame,
    keyword_data: dict | None = None,
) -> dict:
    """Build an independent SIC profile for each channel.

    For each channel, independently compute:
        prevalence(sic, channel) = count(sic, channel) / total_companies_in_channel

    This tells us: "X% of builders merchants have SIC 46730".
    Each channel's profile is self-contained - it doesn't know about
    other channels.

    At prediction time, a channel prior (proportion of total companies)
    is applied so that smaller channels don't dominate larger ones.
    """
    numeric = matrix.drop(columns=["description"], errors="ignore")

    # Build independent profile per channel
    channel_profiles: dict[str, dict[str, float]] = {}
    channel_totals: dict[str, float] = {}
    for channel in numeric.columns:
        col = numeric[channel]
        total = col.sum()
        channel_totals[channel] = total
        if total == 0:
            channel_profiles[channel] = {}
            continue

        profile = {}
        for sic in numeric.index:
            count = float(col[sic])
            if count > 0:
                profile[sic] = count / total  # prevalence in THIS channel
        channel_profiles[channel] = profile

    # Channel priors: proportion of all companies in each channel
    grand_total = sum(channel_totals.values())
    channel_priors: dict[str, float] = {}
    for ch, t in channel_totals.items():
        channel_priors[ch] = t / grand_total if grand_total > 0 else 0.0

    # Also store as sic_weights format for scorer compatibility
    # Convert from {channel: {sic: rate}} to {sic: {channel: rate}}
    all_sics = set()
    for profile in channel_profiles.values():
        all_sics.update(profile.keys())

    sic_weights: dict[str, dict[str, float]] = {}
    for sic in all_sics:
        sic_weights[sic] = {}
        for channel, profile in channel_profiles.items():
            if sic in profile:
                sic_weights[sic][channel] = profile[sic]

    fallback = numeric.sum(axis=0).idxmax()

    rules = {
        "sic_weights": sic_weights,
        "channel_profiles": channel_profiles,
        "channel_priors": channel_priors,
        "fallback": fallback,
    }

    if keyword_data and keyword_data.get("keyword_scores"):
        rules["keyword_scores"] = keyword_data["keyword_scores"]

    # Print channel profiles summary
    print("\n  Channel SIC Profiles (independent):")
    for ch, profile in channel_profiles.items():
        top_sics = sorted(profile.items(), key=lambda x: x[1], reverse=True)[:5]
        top_str = ", ".join(f"{s}({v:.0%})" for s, v in top_sics)
        prior_pct = channel_priors[ch] * 100
        print(f"    {ch} (prior {prior_pct:.1f}%): {len(profile)} SIC codes. Top 5: {top_str}")

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
    """Score a company against each channel's profile independently,
    then weight by the channel prior so smaller channels don't dominate.

    final_score(ch) = prior(ch) * blended_score(ch)

    The prior accounts for the fact that Plumbing Merchants (1039 companies)
    should score higher than Department Stores (12 companies) when SIC codes
    are shared between channels.
    """
    profiles = rules["channel_profiles"]

    # Score against each channel independently
    sic_scores: dict[str, float] = {}
    for channel, profile in profiles.items():
        score = 0.0
        for sic in sic_codes:
            if sic in profile:
                score += profile[sic]
        sic_scores[channel] = score

    # Name keyword scores - only if discovered from data
    kw_scores: dict[str, float] = {}
    if company_name and "keyword_scores" in rules:
        kw_scores = keyword_channel_scores(company_name, rules["keyword_scores"])

    # Blend with channel prior
    all_channels = set(list(sic_scores.keys()) + list(kw_scores.keys()))
    if not all_channels:
        return rules["fallback"]

    priors = rules.get("channel_priors", {})
    use_name = bool(kw_scores) and any(v > 0 for v in kw_scores.values())
    combined: dict[str, float] = {}
    for ch in all_channels:
        s = sic_scores.get(ch, 0.0)
        if use_name:
            k = kw_scores.get(ch, 0.0)
            blended = (1 - name_weight) * s + name_weight * k
        else:
            blended = s
        # Weight by prior so large channels aren't disadvantaged
        combined[ch] = priors.get(ch, 1.0) * blended

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
