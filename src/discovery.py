"""Discover NEW companies from Companies House based on learned channel profiles.

Uses the Companies House advanced search API to find companies by SIC code,
then scores and classifies them into channels.
"""

import json
import time
from pathlib import Path

import pandas as pd

from config import API_BASE_URL, EXCLUDED_SIC_CODES


def search_by_sic(client, sic_code: str, max_results: int = 500) -> list[dict]:
    """Search Companies House for active companies with a specific SIC code.

    Uses the /advanced-search/companies endpoint.
    Returns a list of company dicts.
    """
    results = []
    start_index = 0
    page_size = 100  # max allowed by API

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

        # Stop if we've got all available results
        if start_index >= total_available:
            break

    return results


def discover_new_companies(
    client,
    rules: dict,
    existing_company_numbers: set[str],
    output_path: Path,
    max_per_sic: int = 500,
) -> pd.DataFrame:
    """Search Companies House for new companies matching each channel's profile.

    For each channel, takes the top SIC codes from its profile and searches
    for active companies with those SIC codes. Excludes companies already
    in the input dataset.

    Parameters
    ----------
    client : CompaniesHouseClient
    rules : dict
        Must contain 'channel_profiles' from build_rules().
    existing_company_numbers : set[str]
        Company numbers already in the input Excel (to exclude).
    output_path : Path
        Where to save the discovered companies CSV.
    max_per_sic : int
        Max results to fetch per SIC code search.

    Returns
    -------
    pd.DataFrame with discovered companies and their assigned channel.
    """
    profiles = rules.get("channel_profiles", {})
    if not profiles:
        print("  [ERROR] No channel profiles found in rules.")
        return pd.DataFrame()

    # Collect the SIC codes to search for each channel
    # Use ALL SIC codes in the profile, sorted by prevalence
    channel_sics: dict[str, list[str]] = {}
    for channel, profile in profiles.items():
        sics = sorted(profile.keys(), key=lambda s: profile[s], reverse=True)
        # Remove excluded SIC codes
        sics = [s for s in sics if s not in EXCLUDED_SIC_CODES]
        channel_sics[channel] = sics
        print(f"  {channel}: searching {len(sics)} SIC codes")

    # Search and collect all unique companies
    seen_numbers: set[str] = set(existing_company_numbers)
    all_found: list[dict] = []

    for channel, sics in channel_sics.items():
        print(f"\n  --- Discovering {channel} ---")
        channel_count = 0

        for sic in sics:
            print(f"    Searching SIC {sic} ...", end=" ")
            results = search_by_sic(client, sic, max_results=max_per_sic)
            new_count = 0

            for item in results:
                co_num = item.get("company_number", "")
                if not co_num or co_num in seen_numbers:
                    continue

                seen_numbers.add(co_num)
                new_count += 1

                # Extract address
                addr = item.get("registered_office_address", {})
                address_parts = [
                    addr.get("address_line_1", ""),
                    addr.get("address_line_2", ""),
                    addr.get("locality", ""),
                    addr.get("region", ""),
                ]
                full_address = ", ".join(p for p in address_parts if p)

                sic_codes = item.get("sic_codes", [])
                # Remove excluded SICs
                sic_codes = [s for s in sic_codes if s not in EXCLUDED_SIC_CODES]

                all_found.append({
                    "input_name": "",
                    "matched_name": item.get("company_name", ""),
                    "company_number": co_num,
                    "sic_codes": json.dumps(sic_codes),
                    "company_status": item.get("company_status", "active"),
                    "company_type": item.get("company_type", ""),
                    "date_of_creation": item.get("date_of_creation", ""),
                    "full_address": full_address,
                    "postcode": addr.get("postal_code", ""),
                    "region": addr.get("region", ""),
                    "country": addr.get("country", ""),
                    "channel": "",  # will be assigned by scorer
                    "source_sic": sic,
                    "source_channel": channel,
                })

            print(f"{new_count} new (of {len(results)} total)")
            channel_count += new_count

        print(f"  Total new for {channel}: {channel_count}")

    if not all_found:
        print("  No new companies found.")
        return pd.DataFrame()

    df = pd.DataFrame(all_found)
    df.to_csv(output_path, index=False)
    print(f"\n  Total new companies discovered: {len(df)}")
    print(f"  Saved to {output_path}")
    return df
