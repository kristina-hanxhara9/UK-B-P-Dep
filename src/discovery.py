"""Discover NEW companies from Companies House based on learned channel profiles.

Uses top 4 SIC codes and top 4 keywords per channel to find new companies.
Results are categorised as:
  - SIC matches     : found via SIC code search
  - Keyword matches : found via company name keyword search
  - Combined        : found by BOTH SIC and keyword (highest confidence)
"""

import json
import time
from pathlib import Path

import pandas as pd

from config import API_BASE_URL, EXCLUDED_SIC_CODES

TOP_N = 4  # top 4 SIC codes and top 4 keywords per channel


# ------------------------------------------------------------------
# API search helpers
# ------------------------------------------------------------------

def search_by_sic(client, sic_code: str, max_results: int = 500) -> list[dict]:
    """Search Companies House for active companies with a specific SIC code.

    Uses the /advanced-search/companies endpoint.
    """
    results = []
    start_index = 0
    page_size = 100

    while start_index < max_results:
        client._rate_limit()
        url = f"{API_BASE_URL}/advanced-search/companies"
        params = {
            "sic_codes": sic_code,
            "company_status": "active",
            "size": page_size,
            "start_index": start_index,
        }

        try:
            resp = client.session.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"    [ERROR] Search SIC {sic_code} at offset {start_index}: {exc}")
            break

        items = data.get("items", [])
        if not items:
            break

        results.extend(items)
        total_available = data.get("total_results", 0)
        start_index += page_size

        if start_index >= total_available:
            break

    return results


def search_by_keyword(client, keyword: str, max_results: int = 100) -> list[dict]:
    """Search Companies House for active companies matching a name keyword.

    Uses the /search/companies endpoint.
    """
    results = []
    start_index = 0
    page_size = 50

    while start_index < max_results:
        client._rate_limit()
        url = f"{API_BASE_URL}/search/companies"
        params = {
            "q": keyword,
            "items_per_page": page_size,
            "start_index": start_index,
        }

        try:
            resp = client.session.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"    [ERROR] Search keyword '{keyword}' at offset {start_index}: {exc}")
            break

        items = data.get("items", [])
        if not items:
            break

        # Only keep active companies
        active = [i for i in items if i.get("company_status") == "active"]
        results.extend(active)

        total_available = data.get("total_results", 0)
        start_index += page_size

        if start_index >= total_available:
            break

    return results


# ------------------------------------------------------------------
# Extract company record from API item
# ------------------------------------------------------------------

def _extract_company(item: dict, source_type: str, source_value: str, source_channel: str) -> dict:
    """Convert an API search result item into a flat record."""
    addr = item.get("registered_office_address", {}) or {}
    address_parts = [
        addr.get("address_line_1", ""),
        addr.get("address_line_2", ""),
        addr.get("locality", ""),
        addr.get("region", ""),
    ]
    full_address = ", ".join(p for p in address_parts if p)

    sic_codes = item.get("sic_codes", []) or []
    sic_codes = [s for s in sic_codes if s not in EXCLUDED_SIC_CODES]

    return {
        "input_name": "",
        "matched_name": item.get("company_name", item.get("title", "")),
        "company_number": item.get("company_number", ""),
        "sic_codes": json.dumps(sic_codes),
        "company_status": item.get("company_status", "active"),
        "company_type": item.get("company_type", ""),
        "date_of_creation": item.get("date_of_creation", ""),
        "full_address": full_address,
        "postcode": addr.get("postal_code", ""),
        "region": addr.get("region", ""),
        "country": addr.get("country", ""),
        "channel": "",
        "source_type": source_type,      # "sic", "keyword", or "combined"
        "source_value": source_value,     # the SIC code or keyword used
        "source_channel": source_channel, # which channel profile it came from
    }


# ------------------------------------------------------------------
# Main discovery function
# ------------------------------------------------------------------

def discover_new_companies(
    client,
    rules: dict,
    existing_company_numbers: set[str],
    output_path: Path,
    max_per_sic: int = 500,
    max_per_keyword: int = 100,
) -> dict[str, pd.DataFrame]:
    """Search Companies House for new companies using top 4 SIC codes
    and top 4 keywords per channel.

    Returns a dict with three DataFrames:
      - "sic"      : companies found via SIC code search only
      - "keyword"  : companies found via keyword search only
      - "combined" : companies found by BOTH SIC and keyword searches

    Parameters
    ----------
    client : CompaniesHouseClient
    rules : dict
        Must contain 'channel_profiles' and optionally 'keyword_scores'.
    existing_company_numbers : set[str]
        Company numbers already in the input Excel (to exclude).
    output_path : Path
        Base path for saving CSVs (will create _sic.csv, _keyword.csv, _combined.csv).
    max_per_sic : int
        Max results per SIC code search.
    max_per_keyword : int
        Max results per keyword search.
    """
    profiles = rules.get("channel_profiles", {})
    keyword_scores = rules.get("keyword_scores", {})

    if not profiles:
        print("  [ERROR] No channel profiles found in rules.")
        return {"sic": pd.DataFrame(), "keyword": pd.DataFrame(), "combined": pd.DataFrame()}

    # ----------------------------------------------------------
    # 1. Collect top 4 SIC codes and top 4 keywords per channel
    # ----------------------------------------------------------
    channel_top_sics: dict[str, list[str]] = {}
    channel_top_kws: dict[str, list[str]] = {}

    for channel, profile in profiles.items():
        # Top 4 SIC codes by prevalence
        sics = sorted(profile.keys(), key=lambda s: profile[s], reverse=True)
        sics = [s for s in sics if s not in EXCLUDED_SIC_CODES][:TOP_N]
        channel_top_sics[channel] = sics

        # Top 4 keywords by chi2 score
        kws = keyword_scores.get(channel, [])
        channel_top_kws[channel] = [w for w, _ in kws[:TOP_N]]

    print("\n  Discovery profile per channel:")
    for ch in profiles:
        sics_str = ", ".join(channel_top_sics.get(ch, []))
        kws_str = ", ".join(channel_top_kws.get(ch, []))
        print(f"    {ch}:")
        print(f"      Top {TOP_N} SIC codes: {sics_str}")
        print(f"      Top {TOP_N} keywords:  {kws_str}")

    # ----------------------------------------------------------
    # 2. Search by SIC codes
    # ----------------------------------------------------------
    seen_numbers: set[str] = set(existing_company_numbers)
    sic_found: dict[str, dict] = {}   # company_number -> record

    print("\n  --- SIC Code Searches ---")
    for channel, sics in channel_top_sics.items():
        print(f"\n  {channel}:")
        for sic in sics:
            print(f"    SIC {sic} ...", end=" ")
            results = search_by_sic(client, sic, max_results=max_per_sic)
            new_count = 0

            for item in results:
                co_num = item.get("company_number", "")
                if not co_num or co_num in seen_numbers:
                    continue

                seen_numbers.add(co_num)
                new_count += 1
                sic_found[co_num] = _extract_company(item, "sic", sic, channel)

            print(f"{new_count} new (of {len(results)} total)")

    # ----------------------------------------------------------
    # 3. Search by keywords
    # ----------------------------------------------------------
    # Reset seen so keyword search can find companies also found by SIC
    kw_seen: set[str] = set(existing_company_numbers)
    kw_found: dict[str, dict] = {}   # company_number -> record

    print("\n  --- Keyword Searches ---")
    for channel, kws in channel_top_kws.items():
        if not kws:
            print(f"\n  {channel}: no keywords discovered, skipping")
            continue

        print(f"\n  {channel}:")
        for kw in kws:
            print(f"    Keyword '{kw}' ...", end=" ")
            results = search_by_keyword(client, kw, max_results=max_per_keyword)
            new_count = 0

            for item in results:
                co_num = item.get("company_number", "")
                if not co_num or co_num in kw_seen:
                    continue

                kw_seen.add(co_num)
                new_count += 1
                kw_found[co_num] = _extract_company(item, "keyword", kw, channel)

            print(f"{new_count} new (of {len(results)} total)")

    # ----------------------------------------------------------
    # 4. Categorise: SIC-only, keyword-only, combined (both)
    # ----------------------------------------------------------
    sic_numbers = set(sic_found.keys())
    kw_numbers = set(kw_found.keys())
    combined_numbers = sic_numbers & kw_numbers
    sic_only_numbers = sic_numbers - combined_numbers
    kw_only_numbers = kw_numbers - combined_numbers

    # Build combined records (merge info from both searches)
    combined_records = []
    for co_num in combined_numbers:
        rec = sic_found[co_num].copy()
        rec["source_type"] = "combined"
        # Note both the SIC and keyword that matched
        kw_rec = kw_found[co_num]
        rec["source_value"] = f"SIC:{rec['source_value']} + KW:{kw_rec['source_value']}"
        combined_records.append(rec)

    sic_only_records = [sic_found[n] for n in sic_only_numbers]
    kw_only_records = [kw_found[n] for n in kw_only_numbers]

    # ----------------------------------------------------------
    # 5. Build DataFrames and save
    # ----------------------------------------------------------
    dfs = {}
    base = output_path.parent / output_path.stem

    for label, records in [("sic", sic_only_records), ("keyword", kw_only_records), ("combined", combined_records)]:
        if records:
            df = pd.DataFrame(records)
            csv_path = Path(f"{base}_{label}.csv")
            df.to_csv(csv_path, index=False)
            dfs[label] = df
            print(f"\n  {label.upper()} matches: {len(df)} companies -> {csv_path}")
        else:
            dfs[label] = pd.DataFrame()
            print(f"\n  {label.upper()} matches: 0 companies")

    total = len(sic_only_records) + len(kw_only_records) + len(combined_records)
    print(f"\n  Total new companies discovered: {total}")
    print(f"    SIC-only:     {len(sic_only_records)}")
    print(f"    Keyword-only: {len(kw_only_records)}")
    print(f"    Combined:     {len(combined_records)}")

    return dfs
