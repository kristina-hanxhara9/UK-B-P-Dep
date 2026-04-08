"""Confidence-scored channel predictions with transparent evidence.

Confidence levels
-----------------
HIGH   — Both SIC codes AND name keywords agree on the predicted channel.
MEDIUM — Only one signal (SIC OR keywords) supports the predicted channel.
LOW    — No strong signal, or SIC and keywords disagree on the channel.
"""

import json
from pathlib import Path

import pandas as pd

from src.name_features import tokenize_name, keyword_channel_scores
from src.sic_mapping import SIC_DESCRIPTIONS


# ------------------------------------------------------------------
# Single-company scoring
# ------------------------------------------------------------------

def score_company(
    sic_codes: list[str],
    company_name: str,
    rules: dict,
) -> dict:
    """Score a single company and return prediction + evidence + confidence.

    Returns
    -------
    dict with keys:
        predicted_channel, confidence, combined_score,
        sic_top_channel, sic_scores, top_3_sic,
        kw_top_channel, kw_scores, top_3_keywords
    """
    sic_weights = rules["sic_weights"]

    # --- SIC evidence ---
    sic_scores: dict[str, float] = {}
    sic_detail: list[tuple[str, str, float]] = []  # (sic, channel, weight)

    for sic in sic_codes:
        if sic in sic_weights:
            for ch, w in sic_weights[sic].items():
                sic_scores[ch] = sic_scores.get(ch, 0.0) + w
                sic_detail.append((sic, ch, w))

    sic_top_channel = max(sic_scores, key=sic_scores.get) if sic_scores else None

    # --- Keyword evidence ---
    kw_scores: dict[str, float] = {}
    kw_detail: dict[str, list[tuple[str, float]]] = {}  # channel -> [(word, score)]

    if company_name and "keyword_scores" in rules:
        kw_scores = keyword_channel_scores(company_name, rules["keyword_scores"])

        # Find which specific keywords matched, per channel
        tokens = set(tokenize_name(company_name))
        for ch, scored_words in rules["keyword_scores"].items():
            hits = [(w, s) for w, s in scored_words if w in tokens]
            if hits:
                kw_detail[ch] = sorted(hits, key=lambda x: x[1], reverse=True)

    kw_top_channel = max(kw_scores, key=kw_scores.get) if kw_scores else None

    # --- Blend and predict ---
    has_sic = bool(sic_scores)
    has_kw = bool(kw_scores) and any(v > 0 for v in kw_scores.values())
    name_weight = 0.3

    all_channels = set(list(sic_scores.keys()) + list(kw_scores.keys()))
    if not all_channels:
        predicted = rules["fallback"]
        combined_score = 0.0
    else:
        combined: dict[str, float] = {}
        for ch in all_channels:
            s = sic_scores.get(ch, 0.0)
            k = kw_scores.get(ch, 0.0)
            if has_kw:
                combined[ch] = (1 - name_weight) * s + name_weight * k
            else:
                combined[ch] = s
        predicted = max(combined, key=combined.get)
        combined_score = combined[predicted]

    # --- Confidence ---
    if has_sic and has_kw and sic_top_channel == predicted and kw_top_channel == predicted:
        confidence = "HIGH"
    elif has_sic and sic_top_channel == predicted:
        confidence = "MEDIUM"
    elif has_kw and kw_top_channel == predicted:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # --- Top 3 SIC codes for predicted channel ---
    sic_for_predicted = [
        (sic, w) for sic, ch, w in sic_detail if ch == predicted
    ]
    sic_for_predicted.sort(key=lambda x: x[1], reverse=True)
    top_3_sic = [
        {"sic": sic, "weight": round(w, 3), "description": SIC_DESCRIPTIONS.get(sic, "")}
        for sic, w in sic_for_predicted[:3]
    ]

    # --- Top 3 keywords for predicted channel ---
    kw_for_predicted = kw_detail.get(predicted, [])
    top_3_kw = [
        {"keyword": w, "chi2_score": round(s, 2)}
        for w, s in kw_for_predicted[:3]
    ]

    return {
        "predicted_channel": predicted,
        "confidence": confidence,
        "combined_score": round(combined_score, 4),
        "sic_top_channel": sic_top_channel,
        "sic_scores": {ch: round(v, 4) for ch, v in sic_scores.items()},
        "top_3_sic": top_3_sic,
        "kw_top_channel": kw_top_channel,
        "kw_scores": {ch: round(v, 4) for ch, v in kw_scores.items()},
        "top_3_keywords": top_3_kw,
    }


# ------------------------------------------------------------------
# Batch scoring
# ------------------------------------------------------------------

def score_all_companies(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Score every company in the DataFrame with confidence + evidence.

    Returns a new DataFrame with columns:
        input_name, matched_name, company_number, company_status, channel,
        predicted_channel, confidence, combined_score,
        sic_top_channel, kw_top_channel,
        top_3_sic, top_3_keywords
    """
    results: list[dict] = []

    for _, row in df.iterrows():
        sic = json.loads(row["sic_codes"]) if isinstance(row["sic_codes"], str) else (row["sic_codes"] or [])
        name = str(row.get("matched_name") or row.get("input_name") or "")

        scoring = score_company(sic, name, rules)

        # Format top 3 SIC as readable string
        sic_str = "; ".join(
            f"{s['sic']} ({s['weight']}) {s['description']}"
            for s in scoring["top_3_sic"]
        ) or "(none)"

        # Format top 3 keywords as readable string
        kw_str = "; ".join(
            f"{k['keyword']} ({k['chi2_score']})"
            for k in scoring["top_3_keywords"]
        ) or "(none)"

        results.append({
            "input_name": row.get("input_name", ""),
            "matched_name": row.get("matched_name", ""),
            "company_number": row.get("company_number", ""),
            "company_status": row.get("company_status", ""),
            "company_type": row.get("company_type", ""),
            "date_of_creation": row.get("date_of_creation", ""),
            "full_address": row.get("full_address", ""),
            "postcode": row.get("postcode", ""),
            "locality": row.get("locality", ""),
            "region": row.get("region", ""),
            "country": row.get("country", ""),
            "sic_codes": row.get("sic_codes", "[]"),
            "actual_channel": row.get("channel", ""),
            "predicted_channel": scoring["predicted_channel"],
            "confidence": scoring["confidence"],
            "combined_score": scoring["combined_score"],
            "sic_agrees": scoring["sic_top_channel"],
            "keywords_agree": scoring["kw_top_channel"],
            "top_3_sic_codes": sic_str,
            "top_3_keywords": kw_str,
        })

    return pd.DataFrame(results)


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------

def generate_confidence_report(
    scored_df: pd.DataFrame,
    output_dir: Path,
) -> str:
    """Save scored predictions as CSV and generate a summary report."""
    # Save full details to CSV
    csv_path = output_dir / "scored_predictions.csv"
    scored_df.to_csv(csv_path, index=False)
    print(f"  Saved scored predictions → {csv_path}")

    # Summary
    total = len(scored_df)
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("CONFIDENCE-SCORED PREDICTIONS SUMMARY")
    lines.append("=" * 80)

    # Confidence distribution
    lines.append("\nConfidence Distribution:")
    for level in ["HIGH", "MEDIUM", "LOW"]:
        count = (scored_df["confidence"] == level).sum()
        lines.append(f"  {level:>6s}: {count:>4d} ({count/total:.0%})")

    # Accuracy by confidence level
    if "actual_channel" in scored_df.columns:
        correct = scored_df["predicted_channel"] == scored_df["actual_channel"]
        lines.append("\nAccuracy by Confidence Level:")
        for level in ["HIGH", "MEDIUM", "LOW"]:
            mask = scored_df["confidence"] == level
            if mask.sum() > 0:
                acc = correct[mask].mean()
                lines.append(f"  {level:>6s}: {acc:.0%} ({correct[mask].sum()}/{mask.sum()})")

        lines.append(f"\n  Overall: {correct.mean():.0%} ({correct.sum()}/{total})")

    # Signal agreement breakdown
    lines.append("\nSignal Agreement:")
    both_agree = (
        (scored_df["sic_agrees"] == scored_df["predicted_channel"]) &
        (scored_df["keywords_agree"] == scored_df["predicted_channel"])
    ).sum()
    sic_only = (
        (scored_df["sic_agrees"] == scored_df["predicted_channel"]) &
        (scored_df["keywords_agree"] != scored_df["predicted_channel"])
    ).sum()
    kw_only = (
        (scored_df["sic_agrees"] != scored_df["predicted_channel"]) &
        (scored_df["keywords_agree"] == scored_df["predicted_channel"])
    ).sum()
    neither = total - both_agree - sic_only - kw_only
    lines.append(f"  SIC + Keywords agree: {both_agree}")
    lines.append(f"  SIC only:             {sic_only}")
    lines.append(f"  Keywords only:        {kw_only}")
    lines.append(f"  Neither:              {neither}")

    # Sample predictions at each confidence level
    for level in ["HIGH", "MEDIUM", "LOW"]:
        subset = scored_df[scored_df["confidence"] == level].head(3)
        if subset.empty:
            continue
        lines.append(f"\nSample {level} confidence predictions:")
        for _, row in subset.iterrows():
            lines.append(f"  Company:   {row['input_name']}")
            lines.append(f"  Predicted: {row['predicted_channel']}")
            lines.append(f"  SIC evidence:     {row['top_3_sic_codes']}")
            lines.append(f"  Keyword evidence: {row['top_3_keywords']}")
            lines.append("")

    report = "\n".join(lines)
    report_path = output_dir / "confidence_report.txt"
    report_path.write_text(report)
    print(report)
    print(f"\n  Saved confidence report → {report_path}")
    return report
