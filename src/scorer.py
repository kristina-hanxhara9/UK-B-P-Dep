"""Confidence-scored channel predictions with transparent evidence.

Outputs a multi-sheet Excel file:
  - All Companies       — every company with prediction + confidence + evidence
  - Builders Merchants  — filtered to that predicted channel
  - Plumbing Merchants  — filtered
  - Department Stores   — filtered
  - Top 3 SIC per Channel   — summary of top 3 SIC codes per channel
  - Top 3 KW per Channel    — summary of top 3 keywords per channel
  - SIC + KW Combined       — top 3 of each side by side per channel

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
    """Score a single company and return prediction + evidence + confidence."""
    sic_weights = rules["sic_weights"]

    # --- SIC evidence ---
    sic_scores: dict[str, float] = {}
    sic_detail: list[tuple[str, str, float]] = []

    for sic in sic_codes:
        if sic in sic_weights:
            for ch, w in sic_weights[sic].items():
                sic_scores[ch] = sic_scores.get(ch, 0.0) + w
                sic_detail.append((sic, ch, w))

    sic_top_channel = max(sic_scores, key=sic_scores.get) if sic_scores else None

    # --- Keyword evidence ---
    kw_scores: dict[str, float] = {}
    kw_detail: dict[str, list[tuple[str, float]]] = {}

    if company_name and "keyword_scores" in rules:
        kw_scores = keyword_channel_scores(company_name, rules["keyword_scores"])

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
# Batch scoring → flat DataFrame with separate columns
# ------------------------------------------------------------------

def score_all_companies(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Score every company. Top 3 SIC and top 3 keywords are separate columns."""
    results: list[dict] = []

    for _, row in df.iterrows():
        sic = json.loads(row["sic_codes"]) if isinstance(row["sic_codes"], str) else (row["sic_codes"] or [])
        name = str(row.get("matched_name") or row.get("input_name") or "")

        scoring = score_company(sic, name, rules)

        rec: dict = {
            "input_name": row.get("input_name", ""),
            "matched_name": row.get("matched_name", ""),
            "company_number": row.get("company_number", ""),
            "company_status": row.get("company_status", ""),
            "company_type": row.get("company_type", ""),
            "date_of_creation": row.get("date_of_creation", ""),
            "full_address": row.get("full_address", ""),
            "postcode": row.get("postcode", ""),
            "region": row.get("region", ""),
            "sic_codes": row.get("sic_codes", "[]"),
            "actual_channel": row.get("channel", ""),
            "predicted_channel": scoring["predicted_channel"],
            "confidence": scoring["confidence"],
            "combined_score": scoring["combined_score"],
            "sic_signal": scoring["sic_top_channel"] or "",
            "keyword_signal": scoring["kw_top_channel"] or "",
        }

        # Top 3 SIC codes — separate columns
        for i in range(3):
            if i < len(scoring["top_3_sic"]):
                s = scoring["top_3_sic"][i]
                rec[f"sic_{i+1}_code"] = s["sic"]
                rec[f"sic_{i+1}_weight"] = s["weight"]
                rec[f"sic_{i+1}_description"] = s["description"]
            else:
                rec[f"sic_{i+1}_code"] = ""
                rec[f"sic_{i+1}_weight"] = ""
                rec[f"sic_{i+1}_description"] = ""

        # Top 3 keywords — separate columns
        for i in range(3):
            if i < len(scoring["top_3_keywords"]):
                k = scoring["top_3_keywords"][i]
                rec[f"keyword_{i+1}"] = k["keyword"]
                rec[f"keyword_{i+1}_score"] = k["chi2_score"]
            else:
                rec[f"keyword_{i+1}"] = ""
                rec[f"keyword_{i+1}_score"] = ""

        results.append(rec)

    return pd.DataFrame(results)


# ------------------------------------------------------------------
# Channel summary tables
# ------------------------------------------------------------------

def _build_top3_sic_summary(rules: dict) -> pd.DataFrame:
    """Top 3 SIC codes per channel by weight."""
    rows: list[dict] = []
    sic_weights = rules["sic_weights"]
    channels = set()
    for sic_data in sic_weights.values():
        channels.update(sic_data.keys())

    for ch in sorted(channels):
        # Collect (sic, weight) pairs for this channel
        sic_w = [(sic, data[ch]) for sic, data in sic_weights.items() if data.get(ch, 0) > 0]
        sic_w.sort(key=lambda x: x[1], reverse=True)
        for rank, (sic, w) in enumerate(sic_w[:3], 1):
            rows.append({
                "channel": ch,
                "rank": rank,
                "sic_code": sic,
                "weight": round(w, 3),
                "description": SIC_DESCRIPTIONS.get(sic, ""),
            })

    return pd.DataFrame(rows)


def _build_top3_kw_summary(rules: dict) -> pd.DataFrame:
    """Top 3 keywords per channel by chi2 score."""
    rows: list[dict] = []
    if "keyword_scores" not in rules:
        return pd.DataFrame(columns=["channel", "rank", "keyword", "chi2_score"])

    for ch, scored in rules["keyword_scores"].items():
        for rank, (word, score) in enumerate(scored[:3], 1):
            rows.append({
                "channel": ch,
                "rank": rank,
                "keyword": word,
                "chi2_score": round(score, 2),
            })

    return pd.DataFrame(rows)


def _build_combined_summary(rules: dict) -> pd.DataFrame:
    """Top 3 SIC + top 3 keywords side by side per channel."""
    sic_summary = _build_top3_sic_summary(rules)
    kw_summary = _build_top3_kw_summary(rules)

    channels = sorted(
        set(sic_summary["channel"].unique()) | set(kw_summary["channel"].unique())
    )

    rows: list[dict] = []
    for ch in channels:
        sic_ch = sic_summary[sic_summary["channel"] == ch]
        kw_ch = kw_summary[kw_summary["channel"] == ch]

        for rank in range(1, 4):
            rec = {"channel": ch, "rank": rank}

            sic_row = sic_ch[sic_ch["rank"] == rank]
            if not sic_row.empty:
                r = sic_row.iloc[0]
                rec["sic_code"] = r["sic_code"]
                rec["sic_weight"] = r["weight"]
                rec["sic_description"] = r["description"]
            else:
                rec["sic_code"] = ""
                rec["sic_weight"] = ""
                rec["sic_description"] = ""

            kw_row = kw_ch[kw_ch["rank"] == rank]
            if not kw_row.empty:
                r = kw_row.iloc[0]
                rec["keyword"] = r["keyword"]
                rec["keyword_chi2"] = r["chi2_score"]
            else:
                rec["keyword"] = ""
                rec["keyword_chi2"] = ""

            rows.append(rec)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Multi-sheet Excel output
# ------------------------------------------------------------------

def generate_output_excel(
    scored_df: pd.DataFrame,
    rules: dict,
    output_path: Path,
):
    """Write a multi-sheet Excel file with all results."""
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

        # Sheet 1: All Companies
        scored_df.to_excel(writer, sheet_name="All Companies", index=False)

        # Sheets 2-4: one per predicted channel
        for channel in sorted(scored_df["predicted_channel"].unique()):
            sheet_name = channel[:31]  # Excel sheet name limit
            subset = scored_df[scored_df["predicted_channel"] == channel]
            subset.to_excel(writer, sheet_name=sheet_name, index=False)

        # Sheet 5: Top 3 SIC per Channel
        sic_summary = _build_top3_sic_summary(rules)
        sic_summary.to_excel(writer, sheet_name="Top 3 SIC per Channel", index=False)

        # Sheet 6: Top 3 Keywords per Channel
        kw_summary = _build_top3_kw_summary(rules)
        kw_summary.to_excel(writer, sheet_name="Top 3 KW per Channel", index=False)

        # Sheet 7: Combined SIC + Keywords
        combined = _build_combined_summary(rules)
        combined.to_excel(writer, sheet_name="SIC + KW Combined", index=False)

    print(f"  Saved output Excel → {output_path}")


# ------------------------------------------------------------------
# Text report
# ------------------------------------------------------------------

def generate_confidence_report(
    scored_df: pd.DataFrame,
    rules: dict,
    output_dir: Path,
) -> str:
    """Save scored predictions as Excel + CSV + text summary."""
    # Multi-sheet Excel
    excel_path = output_dir / "results.xlsx"
    generate_output_excel(scored_df, rules, excel_path)

    # CSV backup
    csv_path = output_dir / "scored_predictions.csv"
    scored_df.to_csv(csv_path, index=False)

    # Text summary
    total = len(scored_df)
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("CONFIDENCE-SCORED PREDICTIONS SUMMARY")
    lines.append("=" * 80)

    lines.append("\nConfidence Distribution:")
    for level in ["HIGH", "MEDIUM", "LOW"]:
        count = (scored_df["confidence"] == level).sum()
        pct = f"{count/total:.0%}" if total > 0 else "0%"
        lines.append(f"  {level:>6s}: {count:>4d} ({pct})")

    if "actual_channel" in scored_df.columns:
        correct = scored_df["predicted_channel"] == scored_df["actual_channel"]
        lines.append("\nAccuracy by Confidence Level:")
        for level in ["HIGH", "MEDIUM", "LOW"]:
            mask = scored_df["confidence"] == level
            if mask.sum() > 0:
                acc = correct[mask].mean()
                lines.append(f"  {level:>6s}: {acc:.0%} ({correct[mask].sum()}/{mask.sum()})")
        lines.append(f"\n  Overall: {correct.mean():.0%} ({correct.sum()}/{total})")

    lines.append("\nSignal Agreement:")
    both = (
        (scored_df["sic_signal"] == scored_df["predicted_channel"]) &
        (scored_df["keyword_signal"] == scored_df["predicted_channel"])
    ).sum()
    sic_only = (
        (scored_df["sic_signal"] == scored_df["predicted_channel"]) &
        (scored_df["keyword_signal"] != scored_df["predicted_channel"])
    ).sum()
    kw_only = (
        (scored_df["sic_signal"] != scored_df["predicted_channel"]) &
        (scored_df["keyword_signal"] == scored_df["predicted_channel"])
    ).sum()
    neither = total - both - sic_only - kw_only
    lines.append(f"  SIC + Keywords agree: {both}")
    lines.append(f"  SIC only:             {sic_only}")
    lines.append(f"  Keywords only:        {kw_only}")
    lines.append(f"  Neither:              {neither}")

    # Samples
    for level in ["HIGH", "MEDIUM", "LOW"]:
        subset = scored_df[scored_df["confidence"] == level].head(3)
        if subset.empty:
            continue
        lines.append(f"\nSample {level} confidence predictions:")
        for _, row in subset.iterrows():
            lines.append(f"  Company:     {row['input_name']}")
            lines.append(f"  Predicted:   {row['predicted_channel']} ({row['confidence']})")
            sic_evidence = ", ".join(
                f"{row[f'sic_{i}_code']} ({row[f'sic_{i}_weight']})"
                for i in range(1, 4) if row.get(f"sic_{i}_code")
            ) or "(none)"
            kw_evidence = ", ".join(
                f"{row[f'keyword_{i}']} ({row[f'keyword_{i}_score']})"
                for i in range(1, 4) if row.get(f"keyword_{i}")
            ) or "(none)"
            lines.append(f"  SIC evidence:     {sic_evidence}")
            lines.append(f"  Keyword evidence: {kw_evidence}")
            lines.append("")

    report = "\n".join(lines)
    report_path = output_dir / "confidence_report.txt"
    report_path.write_text(report)
    print(report)
    print(f"\n  Saved confidence report → {report_path}")
    return report
