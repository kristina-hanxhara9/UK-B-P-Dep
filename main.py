#!/usr/bin/env python3
"""UK Merchants Channel Classifier - main pipeline.

Usage
-----
    # Full pipeline (requires API key in .env)
    python main.py --excel data/input/companies.xlsx

    # Skip API calls, re-run analysis + models from cached data
    python main.py --skip-api

    # Run a single step (1=API, 2=SIC mapping, 3=analysis, 4=rule model, 5=ML model, 6=discover)
    python main.py --skip-api --step 3

    # Discover new companies from Companies House (requires API key)
    python main.py --skip-api --step 6
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` imports work from any directory
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd

import config
from src.excel_reader import read_companies
from src.companies_house import CompaniesHouseClient, fetch_all_companies
from src.sic_mapping import build_sic_channel_matrix, print_distribution
from src.analysis import compute_channel_sic_sets, overlap_analysis, unique_sic_codes, shared_sic_codes
from src.name_features import discover_keywords
from src.rule_model import build_rules, evaluate_rule_model
from src.scorer import score_all_companies, generate_output_excel
from src.ml_model import prepare_features, train_and_evaluate
from src.discovery import discover_new_companies
from src.sic_mapping import SIC_DESCRIPTIONS


COMPANY_DATA_CSV = config.PROCESSED_DIR / "company_data.csv"
SIC_MATRIX_CSV = config.PROCESSED_DIR / "sic_channel_matrix.csv"
DISCOVERED_CSV = config.PROCESSED_DIR / "discovered_companies.csv"
RESULTS_EXCEL = config.OUTPUT_DIR / "results.xlsx"
DISCOVERED_EXCEL = config.OUTPUT_DIR / "discovered_companies.xlsx"


import json as _json


def filter_companies(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to active companies only and remove excluded SIC codes.

    This runs BEFORE any analysis so that excluded/inactive companies
    never influence the models or appear in results.
    """
    print("\n" + "=" * 60)
    print("FILTERING - Active only + removing excluded SIC codes")
    print("=" * 60)

    total = len(df)

    # 1. Active only
    df = df[df["company_status"] == "active"].copy()
    active_count = len(df)
    print(f"  Active companies: {active_count} / {total}")

    removed_inactive = total - active_count
    if removed_inactive:
        print(f"  Removed {removed_inactive} non-active companies")

    # 2. Remove excluded SIC codes from each company's sic_codes list
    excluded = config.EXCLUDED_SIC_CODES
    if excluded:
        print(f"  Excluded SIC codes: {', '.join(sorted(excluded))}")

        def _clean_sics(sic_str):
            codes = _json.loads(sic_str) if isinstance(sic_str, str) else (sic_str or [])
            cleaned = [c for c in codes if c not in excluded]
            return _json.dumps(cleaned)

        df["sic_codes"] = df["sic_codes"].apply(_clean_sics)

        # Drop companies that now have zero SIC codes after exclusion
        has_sics = df["sic_codes"].apply(
            lambda x: len(_json.loads(x) if isinstance(x, str) else x) > 0
        )
        dropped = (~has_sics).sum()
        if dropped:
            print(f"  Dropped {dropped} companies with no remaining SIC codes")
            df = df[has_sics]

    print(f"  Final dataset: {len(df)} companies")
    return df.reset_index(drop=True)


def step_1_fetch(excel_path: Path) -> pd.DataFrame:
    """Step 1: Read Excel and fetch data from Companies House."""
    print("\n" + "=" * 60)
    print("STEP 1 - Read Excel & Fetch Companies House Data")
    print("=" * 60)

    if not config.API_KEY:
        print(
            "[ERROR] COMPANIES_HOUSE_API_KEY not set.\n"
            "  Copy .env.example to .env and add your API key.\n"
            "  Get a key at: https://developer.company-information.service.gov.uk/"
        )
        sys.exit(1)

    companies = read_companies(excel_path)
    total = sum(len(v) for v in companies.values())
    print(f"\n  Total companies to look up: {total}")

    client = CompaniesHouseClient(config.API_KEY, config.RAW_CACHE_DIR)
    df = fetch_all_companies(client, companies, COMPANY_DATA_CSV)

    print(f"\n  Status breakdown:")
    print(df["company_status"].value_counts().to_string())
    matched = df["company_number"].notna().sum()
    print(f"\n  Match rate: {matched}/{len(df)} ({matched/len(df):.0%})")

    return df


def step_2_sic_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Build SIC code x channel matrix."""
    print("\n" + "=" * 60)
    print("STEP 2 - SIC Code to Channel Mapping")
    print("=" * 60)

    matrix = build_sic_channel_matrix(df, SIC_MATRIX_CSV)
    print_distribution(matrix)
    return matrix


def step_3_analysis(df: pd.DataFrame, matrix: pd.DataFrame) -> dict:
    """Step 3: Analyse SIC codes AND company name keywords."""
    print("\n" + "=" * 60)
    print("STEP 3 - SIC Code & Keyword Analysis")
    print("=" * 60)

    sic_sets = compute_channel_sic_sets(matrix)
    keyword_data = discover_keywords(df)

    print(f"  SIC codes per channel:")
    for ch, codes in sic_sets.items():
        print(f"    {ch}: {len(codes)} distinct codes")

    print(f"\n  Discovered keywords per channel:")
    for ch, scored in keyword_data.get("keyword_scores", {}).items():
        top_words = [w for w, _ in scored[:5]]
        print(f"    {ch}: {', '.join(top_words)}")

    return keyword_data


def step_4_rule_model(df: pd.DataFrame, matrix: pd.DataFrame, keyword_data: dict | None = None):
    """Step 4: Build and evaluate the rule-based model."""
    print("\n" + "=" * 60)
    print("STEP 4 - Rule-Based Model")
    print("=" * 60)

    rules = build_rules(matrix, keyword_data=keyword_data)
    results = evaluate_rule_model(df, rules)
    return rules, results


def step_5_ml_model(df: pd.DataFrame, keyword_data: dict | None = None):
    """Step 5: Train and evaluate the ML model (active companies only)."""
    print("\n" + "=" * 60)
    print("STEP 5 - Machine Learning Model")
    print("=" * 60)

    discovered_kws = keyword_data.get("all_keywords") if keyword_data else None

    try:
        X, y, mlb = prepare_features(df, discovered_keywords=discovered_kws)
        results = train_and_evaluate(X, y, config.OUTPUT_DIR, mlb=mlb)
        return results
    except ValueError as exc:
        print(f"  [ERROR] Cannot train ML model: {exc}")
        return None


def step_6_discover(df: pd.DataFrame, rules: dict) -> dict[str, pd.DataFrame]:
    """Step 6: Discover NEW companies from Companies House.

    Uses top 4 SIC codes and top 4 keywords per channel.
    Returns dict with 'sic', 'keyword', 'combined' DataFrames.
    """
    print("\n" + "=" * 60)
    print("STEP 6 - Discover New Companies from Companies House")
    print("=" * 60)

    if not config.API_KEY:
        print("[ERROR] COMPANIES_HOUSE_API_KEY required for discovery.")
        return {}

    existing_numbers = set(df["company_number"].dropna().astype(str).tolist())
    print(f"  Excluding {len(existing_numbers)} existing companies")

    client = CompaniesHouseClient(config.API_KEY, config.RAW_CACHE_DIR)
    discovered = discover_new_companies(
        client, rules, existing_numbers, DISCOVERED_CSV
    )
    return discovered


# ------------------------------------------------------------------
# Build the single output Excel with all results
# ------------------------------------------------------------------

def build_analysis_sheets(matrix: pd.DataFrame, keyword_data: dict) -> dict[str, pd.DataFrame]:
    """Build DataFrames for analysis sheets."""
    sheets = {}

    # SIC Distribution sheet
    numeric = matrix.drop(columns=["description"], errors="ignore")
    sic_dist = numeric.copy()
    sic_dist.insert(0, "description", sic_dist.index.map(
        lambda c: SIC_DESCRIPTIONS.get(str(c), "")
    ))
    sic_dist["total"] = numeric.sum(axis=1)
    sic_dist = sic_dist.sort_values("total", ascending=False)
    sic_dist.index.name = "sic_code"
    sheets["SIC Distribution"] = sic_dist.reset_index()

    # SIC Overlap Analysis
    sic_sets = compute_channel_sic_sets(matrix)
    overlaps = overlap_analysis(sic_sets)
    uniques = unique_sic_codes(sic_sets)
    common = shared_sic_codes(sic_sets)

    overlap_rows = []
    for ch, codes in uniques.items():
        for c in sorted(codes):
            overlap_rows.append({
                "type": f"Unique to {ch}",
                "sic_code": c,
                "description": SIC_DESCRIPTIONS.get(c, ""),
            })
    for c in sorted(common):
        overlap_rows.append({
            "type": "Shared (all channels)",
            "sic_code": c,
            "description": SIC_DESCRIPTIONS.get(c, ""),
        })
    for key, val in overlaps.items():
        if key == "all_channels":
            continue
        for c in val["codes"]:
            overlap_rows.append({
                "type": f"Overlap: {key}",
                "sic_code": c,
                "description": SIC_DESCRIPTIONS.get(c, ""),
            })
    sheets["SIC Overlap Analysis"] = pd.DataFrame(overlap_rows)

    # Keyword Analysis
    kw_rows = []
    for ch, scored in keyword_data.get("keyword_scores", {}).items():
        for rank, (word, score) in enumerate(scored, 1):
            kw_rows.append({
                "channel": ch,
                "rank": rank,
                "keyword": word,
                "chi2_score": round(score, 2),
            })
    sheets["Keyword Analysis"] = pd.DataFrame(kw_rows)

    return sheets


def build_model_sheets(rule_results: dict, ml_results: dict | None) -> dict[str, pd.DataFrame]:
    """Build DataFrames for model evaluation sheets."""
    sheets = {}

    # Rule model results
    if rule_results:
        rule_rows = []
        report_lines = rule_results.get("classification_report", "").strip().split("\n")
        for line in report_lines:
            line = line.strip()
            if line and not line.startswith("accuracy") and not line.startswith("macro") and not line.startswith("weighted"):
                parts = line.split()
                if len(parts) >= 5:
                    # Channel name might be multiple words
                    # Find numeric values from the end
                    nums = []
                    words = []
                    for p in reversed(parts):
                        try:
                            nums.insert(0, float(p))
                        except ValueError:
                            words.insert(0, p)
                    if len(nums) >= 4:
                        rule_rows.append({
                            "channel": " ".join(words),
                            "precision": nums[0],
                            "recall": nums[1],
                            "f1_score": nums[2],
                            "support": int(nums[3]),
                        })

            if line.startswith("accuracy"):
                parts = line.split()
                try:
                    acc = float(parts[1])
                    rule_rows.append({
                        "channel": "OVERALL ACCURACY",
                        "precision": acc,
                        "recall": acc,
                        "f1_score": acc,
                        "support": int(parts[2]) if len(parts) > 2 else "",
                    })
                except (ValueError, IndexError):
                    pass

        if rule_rows:
            sheets["Rule Model Results"] = pd.DataFrame(rule_rows)

    # ML model results
    if ml_results:
        ml_rows = []
        cv = ml_results.get("cv_results", {})
        for metric in ["accuracy", "f1_macro", "precision_macro", "recall_macro"]:
            key = f"test_{metric}"
            if key in cv:
                vals = cv[key]
                ml_rows.append({
                    "metric": metric,
                    "mean": round(vals.mean(), 3),
                    "std": round(vals.std(), 3),
                })
        if ml_rows:
            sheets["ML Model Results"] = pd.DataFrame(ml_rows)

    return sheets


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UK Merchants Channel Classifier Pipeline"
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=config.EXCEL_PATH,
        help="Path to the input Excel file (default: data/input/companies.xlsx)",
    )
    parser.add_argument(
        "--step",
        choices=["all", "1", "2", "3", "4", "5", "6"],
        default="all",
        help="Run a specific step or 'all' (default: all). Step 6 = discover new companies.",
    )
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Skip API calls; load company data from cached CSV instead",
    )
    args = parser.parse_args()

    # ----------------------------------------------------------
    # Load or fetch company data
    # ----------------------------------------------------------
    if args.step in ("all", "1") and not args.skip_api:
        df_raw = step_1_fetch(args.excel)
    elif COMPANY_DATA_CSV.exists():
        print(f"\n  Loading cached company data from {COMPANY_DATA_CSV}")
        df_raw = pd.read_csv(COMPANY_DATA_CSV)
    else:
        print(
            "[ERROR] No cached data found. Run step 1 first "
            "(without --skip-api) to fetch company data."
        )
        sys.exit(1)

    # ----------------------------------------------------------
    # Filter: active only + remove excluded SIC codes
    # ----------------------------------------------------------
    df = filter_companies(df_raw)

    # ----------------------------------------------------------
    # Steps 2-5
    # ----------------------------------------------------------
    if args.step in ("all", "2"):
        matrix = step_2_sic_mapping(df)
    elif SIC_MATRIX_CSV.exists():
        matrix = pd.read_csv(SIC_MATRIX_CSV, index_col=0)
    else:
        matrix = step_2_sic_mapping(df)

    keyword_data = None
    if args.step in ("all", "3", "6"):
        keyword_data = step_3_analysis(df, matrix)

    rules = None
    rule_results = None
    if args.step in ("all", "4", "6"):
        rules, rule_results = step_4_rule_model(df, matrix, keyword_data=keyword_data)

    ml_results = None
    if args.step in ("all", "5"):
        ml_results = step_5_ml_model(df, keyword_data=keyword_data)

    # ----------------------------------------------------------
    # Build single output Excel with ALL results (input companies)
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("GENERATING OUTPUT EXCEL")
    print("=" * 60)

    extra_sheets = {}

    # Analysis sheets
    if keyword_data:
        extra_sheets.update(build_analysis_sheets(matrix, keyword_data))

    # Model evaluation sheets
    if rule_results or ml_results:
        extra_sheets.update(build_model_sheets(rule_results or {}, ml_results))

    # Scored predictions + everything into one Excel
    if rules:
        scored_df = score_all_companies(df, rules)

        # Add ML model prediction alongside rule-based prediction
        if ml_results and "predictions" in ml_results:
            ml_preds = ml_results["predictions"]
            scored_df["ml_predicted_channel"] = scored_df.index.map(
                ml_preds
            ).fillna("")
            scored_df["models_agree"] = (
                scored_df["predicted_channel"] == scored_df["ml_predicted_channel"]
            ).map({True: "Yes", False: "No"})
            scored_df.loc[scored_df["ml_predicted_channel"] == "", "models_agree"] = ""

        # Show actual vs predicted distribution
        print("\n  Input (actual) distribution:")
        print(scored_df["actual_channel"].value_counts().to_string())
        print("\n  Predicted distribution:")
        print(scored_df["predicted_channel"].value_counts().to_string())
        print("\n  Correctly classified:")
        correct = scored_df["predicted_channel"] == scored_df["actual_channel"]
        for ch in sorted(scored_df["actual_channel"].unique()):
            mask = scored_df["actual_channel"] == ch
            ch_correct = correct[mask].sum()
            ch_total = mask.sum()
            print(f"    {ch}: {ch_correct}/{ch_total} ({ch_correct/ch_total:.0%})")

        generate_output_excel(scored_df, rules, RESULTS_EXCEL, extra_sheets=extra_sheets)
    else:
        # No rules built yet - just output raw data
        with pd.ExcelWriter(RESULTS_EXCEL, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="All Companies", index=False)
            for name, sheet_df in extra_sheets.items():
                sheet_df.to_excel(writer, sheet_name=name[:31], index=False)
        print(f"  Saved output -> {RESULTS_EXCEL}")

    # ----------------------------------------------------------
    # Step 6 - Discover new companies from Companies House
    # ----------------------------------------------------------
    discovered = None
    if args.step in ("all", "6") and rules:
        discovered = step_6_discover(df, rules)

        has_results = discovered and any(
            not v.empty for v in discovered.values() if isinstance(v, pd.DataFrame)
        )

        if has_results:
            print("\n" + "=" * 60)
            print("WRITING DISCOVERED COMPANIES")
            print("=" * 60)

            # Output columns (raw data, no scoring)
            out_cols = [
                "matched_name", "company_number", "company_status",
                "company_type", "date_of_creation",
                "sic_codes", "full_address", "postcode", "region",
                "source_channel", "source_type", "source_value",
            ]

            with pd.ExcelWriter(DISCOVERED_EXCEL, engine="openpyxl") as writer:
                all_records = []

                for label, label_name in [
                    ("sic", "SIC Matches"),
                    ("keyword", "Keyword Matches"),
                    ("combined", "Combined Matches"),
                ]:
                    disc_df = discovered.get(label, pd.DataFrame())
                    if disc_df.empty:
                        print(f"  {label_name}: 0 companies")
                        continue

                    disc_df["match_type"] = label
                    all_records.append(disc_df)

                    # Per-channel sheets for this match type
                    channels = sorted(disc_df["source_channel"].unique())
                    for ch in channels:
                        sheet_name = f"{label[:3]}_{ch}"[:31]
                        subset = disc_df[disc_df["source_channel"] == ch]
                        avail = [c for c in out_cols if c in subset.columns]
                        subset[avail].to_excel(writer, sheet_name=sheet_name, index=False)
                        print(f"    {sheet_name}: {len(subset)} companies")

                    print(f"  {label_name}: {len(disc_df)} total")

                # Summary sheet with ALL discovered companies
                if all_records:
                    summary = pd.concat(all_records, ignore_index=True)
                    summary_cols = out_cols + ["match_type"]
                    avail = [c for c in summary_cols if c in summary.columns]
                    summary[avail].to_excel(writer, sheet_name="All Discovered", index=False)
                    print(f"\n  TOTAL discovered: {len(summary)}")

            print(f"  Saved -> {DISCOVERED_EXCEL}")

    elif args.step == "6" and not rules:
        print("\n  [ERROR] Rules required for discovery. Run steps 2-4 first.")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    output_files = [str(RESULTS_EXCEL)]
    if discovered and any(not v.empty for v in discovered.values() if isinstance(v, pd.DataFrame)):
        output_files.append(str(DISCOVERED_EXCEL))
    print("DONE - Output files:")
    for f in output_files:
        print(f"  {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
