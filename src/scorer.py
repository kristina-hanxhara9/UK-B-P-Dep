"""Confidence-scored channel predictions with transparent evidence.

Outputs a multi-sheet Excel file:
  - All Companies       -every company with prediction + confidence + evidence
  - Builders Merchants  -filtered to that predicted channel
  - Plumbing Merchants  -filtered
  - Department Stores   -filtered
  - Top 3 SIC per Channel   -summary of top 3 SIC codes per channel
  - Top 3 KW per Channel    -summary of top 3 keywords per channel
  - SIC + KW Combined       -top 3 of each side by side per channel

Confidence levels
-----------------
HIGH   -Both SIC codes AND name keywords agree on the predicted channel.
MEDIUM -Only one signal (SIC OR keywords) supports the predicted channel.
LOW    -No strong signal, or SIC and keywords disagree on the channel.
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
    """Score a company against each channel's independent profile,
    weighted by channel prior so smaller channels don't dominate."""
    profiles = rules.get("channel_profiles", {})
    sic_weights = rules["sic_weights"]

    # --- SIC evidence: score against each channel independently ---
    sic_scores: dict[str, float] = {}
    sic_detail: list[tuple[str, str, float]] = []

    for channel, profile in profiles.items():
        score = 0.0
        for sic in sic_codes:
            if sic in profile:
                w = profile[sic]
                score += w
                sic_detail.append((sic, channel, w))
        sic_scores[channel] = score

    # Fallback: use sic_weights if no channel_profiles
    if not profiles:
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

    # --- Blend and predict (with channel prior) ---
    has_sic = bool(sic_scores)
    has_kw = bool(kw_scores) and any(v > 0 for v in kw_scores.values())
    name_weight = 0.3
    priors = rules.get("channel_priors", {})

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
                blended = (1 - name_weight) * s + name_weight * k
            else:
                blended = s
            combined[ch] = priors.get(ch, 1.0) * blended
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
# Batch scoring ->flat DataFrame with separate columns
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

        # Top 3 SIC codes -separate columns
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

        # Top 3 keywords -separate columns
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
    """Top 3 SIC codes per channel from independent channel profiles."""
    rows: list[dict] = []
    profiles = rules.get("channel_profiles", {})

    for ch in sorted(profiles.keys()):
        profile = profiles[ch]
        top_sics = sorted(profile.items(), key=lambda x: x[1], reverse=True)[:3]
        for rank, (sic, prevalence) in enumerate(top_sics, 1):
            rows.append({
                "channel": ch,
                "rank": rank,
                "sic_code": sic,
                "prevalence": f"{prevalence:.0%}",
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
    """Top 3 SIC codes + top 3 keywords side by side per channel."""
    sic_summary = _build_top3_sic_summary(rules)
    kw_summary = _build_top3_kw_summary(rules)

    channels = sorted(
        set(sic_summary["channel"].unique()) | set(kw_summary["channel"].unique())
    )

    rows: list[dict] = []
    for ch in channels:
        sic_ch = sic_summary[sic_summary["channel"] == ch]
        kw_ch = kw_summary[kw_summary["channel"] == ch]
        max_rank = max(len(sic_ch), len(kw_ch))

        for rank in range(1, max_rank + 1):
            rec = {"channel": ch, "rank": rank}

            sic_row = sic_ch[sic_ch["rank"] == rank]
            if not sic_row.empty:
                r = sic_row.iloc[0]
                rec["sic_code"] = r["sic_code"]
                rec["sic_prevalence"] = r["prevalence"]
                rec["sic_description"] = r["description"]
            else:
                rec["sic_code"] = ""
                rec["sic_prevalence"] = ""
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
    extra_sheets: dict[str, pd.DataFrame] | None = None,
):
    """Write a multi-sheet Excel file with all results.

    Companies are split into per-channel sheets (no duplicates).
    Each company appears in exactly one channel sheet based on its
    predicted_channel.
    """
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

        # Per-channel sheets - each company appears in ONE sheet only
        channels = sorted(scored_df["predicted_channel"].unique())
        for channel in channels:
            sheet_name = channel[:31]
            subset = scored_df[scored_df["predicted_channel"] == channel]
            subset.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"    {channel}: {len(subset)} companies")

        # Summary sheet with all companies (compact: key columns only)
        summary_cols = [
            "input_name", "matched_name", "company_number", "company_status",
            "predicted_channel", "confidence",
            "sic_1_code", "sic_1_description",
            "keyword_1", "keyword_2", "keyword_3",
        ]
        available_cols = [c for c in summary_cols if c in scored_df.columns]
        scored_df[available_cols].to_excel(writer, sheet_name="Summary (all)", index=False)

        # Top 3 SIC per Channel
        sic_summary = _build_top3_sic_summary(rules)
        sic_summary.to_excel(writer, sheet_name="Top 3 SIC per Channel", index=False)

        # Top 3 Keywords per Channel
        kw_summary = _build_top3_kw_summary(rules)
        kw_summary.to_excel(writer, sheet_name="Top 3 KW per Channel", index=False)

        # Combined SIC + Keywords side by side
        combined = _build_combined_summary(rules)
        combined.to_excel(writer, sheet_name="SIC + KW Combined", index=False)

        # Extra sheets from analysis, model evaluation, etc.
        if extra_sheets:
            for name, sheet_df in extra_sheets.items():
                sheet_df.to_excel(writer, sheet_name=name[:31], index=False)

    print(f"  Saved output Excel -> {output_path}")


