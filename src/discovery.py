"""Discover NEW companies from Companies House based on learned channel profiles.

For each channel, uses its top 4 SIC codes and top 4 keywords to pull
new companies. Results are organised into three sheets per channel:
  - SIC sheet     : companies found searching by SIC codes
  - Keyword sheet : companies found searching by keywords
  - Combined sheet: companies found searching by SIC codes FILTERED
                    to only those whose name also contains a channel keyword
"""

import json
from pathlib import Path

import pandas as pd

from config import API_BASE_URL, EXCLUDED_SIC_CODES

TOP_N = 4  # top 4 SIC codes and top 4 keywords per channel


# ------------------------------------------------------------------
# API search helpers
# ------------------------------------------------------------------

def search_by_sic(client, sic_code: str, max_results: int = 5000) -> list[dict]:
    """Search Companies House for active companies with a specific SIC code."""
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
            if resp.status_code != 200:
                print(f"\n      [WARN] SIC {sic_code} returned HTTP {resp.status_code}")
                break
            data = resp.json()
        except Exception as exc:
            print(f"\n      [ERROR] SIC {sic_code}: {exc}")
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


def search_by_keyword(client, keyword: str, max_results: int = 500) -> list[dict]:
    """Search Companies House for active companies by name keyword."""
    results = []
    start_index = 0
    page_size = 100

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
            if resp.status_code != 200:
                print(f"\n      [WARN] Keyword '{keyword}' returned HTTP {resp.status_code}")
                break
            data = resp.json()
        except Exception as exc:
            print(f"\n      [ERROR] Keyword '{keyword}': {exc}")
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

def _extract_record(item: dict, source_channel: str, source_type: str, source_value: str) -> dict:
    """Convert an API result into a flat row."""
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
        "matched_name": item.get("company_name", item.get("title", "")),
        "company_number": item.get("company_number", ""),
        "sic_codes": json.dumps(sic_codes),
        "company_status": item.get("company_status", "active"),
        "company_type": item.get("company_type", ""),
        "date_of_creation": item.get("date_of_creation", ""),
        "full_address": full_address,
        "postcode": addr.get("postal_code", ""),
        "region": addr.get("region", ""),
        "source_channel": source_channel,
        "source_type": source_type,
        "source_value": source_value,
    }


# ------------------------------------------------------------------
# Main discovery
# ------------------------------------------------------------------

def discover_new_companies(
    client,
    rules: dict,
    existing_company_numbers: set[str],
    output_path: Path,
) -> dict[str, pd.DataFrame]:
    """For each channel, search by top 4 SIC codes and top 4 keywords.

    Returns dict with three DataFrames: 'sic', 'keyword', 'combined'.
    Each channel's results are independent (no cross-channel dedup).
    """
    profiles = rules.get("channel_profiles", {})
    keyword_scores = rules.get("keyword_scores", {})

    if not profiles:
        print("  [ERROR] No channel profiles in rules.")
        return {"sic": pd.DataFrame(), "keyword": pd.DataFrame(), "combined": pd.DataFrame()}

    # ----------------------------------------------------------
    # 1. Get top 4 SIC codes + top 4 keywords per channel
    # ----------------------------------------------------------
    channel_top_sics: dict[str, list[str]] = {}
    channel_top_kws: dict[str, list[str]] = {}

    for channel, profile in profiles.items():
        sics = sorted(profile.keys(), key=lambda s: profile[s], reverse=True)
        sics = [s for s in sics if s not in EXCLUDED_SIC_CODES][:TOP_N]
        channel_top_sics[channel] = sics

        kws = keyword_scores.get(channel, [])
        channel_top_kws[channel] = [w for w, _ in kws[:TOP_N]]

    print("\n  Discovery profile per channel:")
    for ch in profiles:
        print(f"    {ch}:")
        print(f"      Top {TOP_N} SIC codes: {', '.join(channel_top_sics.get(ch, []))}")
        print(f"      Top {TOP_N} keywords:  {', '.join(channel_top_kws.get(ch, []))}")

    sic_all_records: list[dict] = []
    kw_all_records: list[dict] = []
    combined_all_records: list[dict] = []

    # ----------------------------------------------------------
    # 2. For EACH channel independently
    # ----------------------------------------------------------
    for channel in profiles:
        top_sics = channel_top_sics[channel]
        top_kws = channel_top_kws[channel]
        kw_set = set(w.lower() for w in top_kws)

        # Track per-channel to avoid duplicates within same channel
        ch_sic_seen: set[str] = set(existing_company_numbers)
        ch_kw_seen: set[str] = set(existing_company_numbers)

        # --- SIC code searches ---
        print(f"\n  === {channel} - SIC Code Searches ===")
        ch_sic_records: list[dict] = []
        ch_combined_records: list[dict] = []

        for sic in top_sics:
            print(f"    SIC {sic} ...", end=" ", flush=True)
            results = search_by_sic(client, sic)
            new_count = 0

            for item in results:
                co_num = item.get("company_number", "")
                if not co_num or co_num in ch_sic_seen:
                    continue
                ch_sic_seen.add(co_num)
                new_count += 1

                rec = _extract_record(item, channel, "sic", sic)

                # Check if company name also contains a channel keyword
                name_lower = rec["matched_name"].lower()
                matched_kws = [k for k in kw_set if k in name_lower]

                if matched_kws:
                    combo = rec.copy()
                    combo["source_type"] = "combined"
                    combo["source_value"] = f"SIC:{sic} + KW:{','.join(matched_kws)}"
                    ch_combined_records.append(combo)

                ch_sic_records.append(rec)

            print(f"{new_count} new (of {len(results)} total)")

        # --- Keyword searches ---
        print(f"\n  === {channel} - Keyword Searches ===")
        ch_kw_records: list[dict] = []

        for kw in top_kws:
            print(f"    Keyword '{kw}' ...", end=" ", flush=True)
            results = search_by_keyword(client, kw)
            new_count = 0

            for item in results:
                co_num = item.get("company_number", "")
                if not co_num or co_num in ch_kw_seen:
                    continue
                ch_kw_seen.add(co_num)
                new_count += 1
                ch_kw_records.append(_extract_record(item, channel, "keyword", kw))

            print(f"{new_count} new (of {len(results)} total)")

        # --- Channel summary ---
        print(f"\n  {channel} totals: SIC={len(ch_sic_records)}, "
              f"Keyword={len(ch_kw_records)}, Combined={len(ch_combined_records)}")

        sic_all_records.extend(ch_sic_records)
        kw_all_records.extend(ch_kw_records)
        combined_all_records.extend(ch_combined_records)

    # ----------------------------------------------------------
    # 3. Build DataFrames and save CSVs
    # ----------------------------------------------------------
    dfs: dict[str, pd.DataFrame] = {}
    base = output_path.parent / output_path.stem

    for label, records in [("sic", sic_all_records), ("keyword", kw_all_records), ("combined", combined_all_records)]:
        if records:
            frame = pd.DataFrame(records)
            csv_path = Path(f"{base}_{label}.csv")
            frame.to_csv(csv_path, index=False)
            dfs[label] = frame
            print(f"\n  {label.upper()}: {len(frame)} companies -> {csv_path}")
        else:
            dfs[label] = pd.DataFrame()
            print(f"\n  {label.upper()}: 0 companies")

    grand_total = len(sic_all_records) + len(kw_all_records) + len(combined_all_records)
    print(f"\n  GRAND TOTAL: {grand_total} records")

    return dfs
