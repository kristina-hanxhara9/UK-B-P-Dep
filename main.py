#!/usr/bin/env python3
"""UK Merchants Channel Classifier — main pipeline.

Usage
-----
    # Full pipeline (requires API key in .env)
    python main.py --excel data/input/companies.xlsx

    # Skip API calls, re-run analysis + models from cached data
    python main.py --skip-api

    # Run a single step (1=API, 2=SIC mapping, 3=analysis, 4=rule model, 5=ML model)
    python main.py --skip-api --step 3
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

import config
from src.excel_reader import read_companies
from src.companies_house import CompaniesHouseClient, fetch_all_companies
from src.sic_mapping import build_sic_channel_matrix, print_distribution
from src.analysis import (
    compute_channel_sic_sets,
    generate_analysis_report,
    plot_sic_heatmap,
)
from src.rule_model import build_rules, evaluate_rule_model
from src.ml_model import prepare_features, train_and_evaluate


COMPANY_DATA_CSV = config.PROCESSED_DIR / "company_data.csv"
SIC_MATRIX_CSV = config.PROCESSED_DIR / "sic_channel_matrix.csv"


def step_1_fetch(excel_path: Path) -> pd.DataFrame:
    """Step 1: Read Excel and fetch data from Companies House."""
    print("\n" + "=" * 60)
    print("STEP 1 — Read Excel & Fetch Companies House Data")
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

    # Summary
    print(f"\n  Status breakdown:")
    print(df["company_status"].value_counts().to_string())
    matched = df["company_number"].notna().sum()
    print(f"\n  Match rate: {matched}/{len(df)} ({matched/len(df):.0%})")

    return df


def step_2_sic_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Build SIC code × channel matrix."""
    print("\n" + "=" * 60)
    print("STEP 2 — SIC Code to Channel Mapping")
    print("=" * 60)

    matrix = build_sic_channel_matrix(df, SIC_MATRIX_CSV)
    print_distribution(matrix, config.OUTPUT_DIR / "sic_distribution.txt")
    return matrix


def step_3_analysis(matrix: pd.DataFrame):
    """Step 3: Analyse SIC code overlaps and uniqueness."""
    print("\n" + "=" * 60)
    print("STEP 3 — SIC Code Analysis")
    print("=" * 60)

    sic_sets = compute_channel_sic_sets(matrix)
    generate_analysis_report(matrix, sic_sets, config.OUTPUT_DIR / "analysis_report.txt")

    try:
        plot_sic_heatmap(matrix, config.OUTPUT_DIR / "sic_heatmap.png")
    except Exception as exc:
        print(f"  [WARN] Could not generate heatmap: {exc}")

    return sic_sets


def step_4_rule_model(df: pd.DataFrame, matrix: pd.DataFrame):
    """Step 4: Build and evaluate the rule-based model."""
    print("\n" + "=" * 60)
    print("STEP 4 — Rule-Based Model")
    print("=" * 60)

    rules = build_rules(matrix)
    results = evaluate_rule_model(df, rules, config.OUTPUT_DIR / "rule_model_results.txt")
    return rules, results


def step_5_ml_model(df: pd.DataFrame):
    """Step 5: Train and evaluate the ML model (active companies only)."""
    print("\n" + "=" * 60)
    print("STEP 5 — Machine Learning Model")
    print("=" * 60)

    try:
        X, y, mlb = prepare_features(df)
        results = train_and_evaluate(X, y, config.OUTPUT_DIR)
        return results
    except ValueError as exc:
        print(f"  [ERROR] Cannot train ML model: {exc}")
        return None


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
        choices=["all", "1", "2", "3", "4", "5"],
        default="all",
        help="Run a specific step or 'all' (default: all)",
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
        df = step_1_fetch(args.excel)
    elif COMPANY_DATA_CSV.exists():
        print(f"\n  Loading cached company data from {COMPANY_DATA_CSV}")
        df = pd.read_csv(COMPANY_DATA_CSV)
    else:
        print(
            "[ERROR] No cached data found. Run step 1 first "
            "(without --skip-api) to fetch company data."
        )
        sys.exit(1)

    # ----------------------------------------------------------
    # Steps 2–5
    # ----------------------------------------------------------
    if args.step in ("all", "2"):
        matrix = step_2_sic_mapping(df)
    elif SIC_MATRIX_CSV.exists():
        matrix = pd.read_csv(SIC_MATRIX_CSV, index_col=0)
    else:
        matrix = step_2_sic_mapping(df)

    if args.step in ("all", "3"):
        step_3_analysis(matrix)

    if args.step in ("all", "4"):
        step_4_rule_model(df, matrix)

    if args.step in ("all", "5"):
        step_5_ml_model(df)

    print("\n" + "=" * 60)
    print("DONE — Check data/output/ for all results")
    print("=" * 60)


if __name__ == "__main__":
    main()
