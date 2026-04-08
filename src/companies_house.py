"""Companies House API client with rate limiting and caching."""

import json
import re
import time
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import requests

from config import API_BASE_URL, RATE_LIMIT_CALLS, RATE_LIMIT_WINDOW


class CompaniesHouseClient:
    """Client for the UK Companies House REST API."""

    def __init__(self, api_key: str, cache_dir: Path):
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # HTTP Basic Auth: API key as username, empty password
        self.session = requests.Session()
        self.session.auth = (api_key, "")
        self.session.headers.update({"Accept": "application/json"})

        # Rate limiter -track timestamps of recent calls
        self._call_times: deque[float] = deque()

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _rate_limit(self):
        """Block until we are within the rate limit window."""
        now = time.monotonic()

        # Purge timestamps older than the window
        while self._call_times and (now - self._call_times[0]) > RATE_LIMIT_WINDOW:
            self._call_times.popleft()

        if len(self._call_times) >= RATE_LIMIT_CALLS:
            wait = RATE_LIMIT_WINDOW - (now - self._call_times[0]) + 1
            print(f"  [RATE LIMIT] Sleeping {wait:.0f}s ...")
            time.sleep(wait)

        self._call_times.append(time.monotonic())

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        return re.sub(r'[^\w\-]', '_', name.lower()).strip('_')[:200]

    def _cache_path(self, company_name: str) -> Path:
        return self.cache_dir / f"{self._sanitize_filename(company_name)}.json"

    def _load_cache(self, company_name: str) -> dict | None:
        path = self._cache_path(company_name)
        if path.exists():
            return json.loads(path.read_text())
        return None

    def _save_cache(self, company_name: str, data: dict):
        path = self._cache_path(company_name)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------

    def search_company(self, query: str) -> list[dict]:
        """Search for companies by name. Returns up to 5 results."""
        self._rate_limit()
        url = f"{API_BASE_URL}/search/companies"
        resp = self.session.get(url, params={"q": query, "items_per_page": 5})
        resp.raise_for_status()
        return resp.json().get("items", [])

    def get_company_profile(self, company_number: str) -> dict:
        """Fetch full company profile by company number."""
        self._rate_limit()
        url = f"{API_BASE_URL}/company/{company_number}"
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Lookup (search ->best match ->profile ->cache)
    # ------------------------------------------------------------------

    def lookup_company(self, company_name: str) -> dict | None:
        """Look up a company: check cache, else search + profile.

        Returns a dict with keys:
            input_name, matched_name, company_number, sic_codes, company_status
        or None if no match found.
        """
        cached = self._load_cache(company_name)
        if cached is not None:
            return cached

        try:
            results = self.search_company(company_name)
        except requests.HTTPError as exc:
            print(f"  [ERROR] Search failed for '{company_name}': {exc}")
            return None

        if not results:
            print(f"  [MISS] No results for '{company_name}'")
            return None

        # Pick the best match using fuzzy string similarity
        best = max(
            results,
            key=lambda r: SequenceMatcher(
                None,
                company_name.lower(),
                r.get("title", "").lower(),
            ).ratio(),
        )

        company_number = best.get("company_number")
        if not company_number:
            return None

        try:
            profile = self.get_company_profile(company_number)
        except requests.HTTPError as exc:
            print(f"  [ERROR] Profile failed for '{company_name}' ({company_number}): {exc}")
            return None

        # Extract registered office address
        addr = profile.get("registered_office_address", {})
        address_parts = [
            addr.get("address_line_1", ""),
            addr.get("address_line_2", ""),
            addr.get("locality", ""),
            addr.get("region", ""),
        ]
        full_address = ", ".join(p for p in address_parts if p)

        data = {
            "input_name": company_name,
            "matched_name": profile.get("company_name", best.get("title", "")),
            "company_number": company_number,
            "sic_codes": profile.get("sic_codes", []),
            "company_status": profile.get("company_status", "unknown"),
            "company_type": profile.get("type", ""),
            "date_of_creation": profile.get("date_of_creation", ""),
            "address_line_1": addr.get("address_line_1", ""),
            "address_line_2": addr.get("address_line_2", ""),
            "locality": addr.get("locality", ""),
            "region": addr.get("region", ""),
            "postcode": addr.get("postal_code", ""),
            "country": addr.get("country", ""),
            "full_address": full_address,
        }

        self._save_cache(company_name, data)
        return data


# ------------------------------------------------------------------
# Batch fetch
# ------------------------------------------------------------------


def fetch_all_companies(
    client: CompaniesHouseClient,
    companies_by_channel: dict[str, list[str]],
    output_path: Path,
) -> pd.DataFrame:
    """Fetch Companies House data for every company in every channel.

    Returns a DataFrame and saves it to *output_path* as CSV.
    """
    rows: list[dict] = []
    total = sum(len(v) for v in companies_by_channel.values())
    done = 0

    for channel, names in companies_by_channel.items():
        for name in names:
            done += 1
            result = client.lookup_company(name)
            if result is None:
                rows.append({
                    "input_name": name,
                    "matched_name": None,
                    "company_number": None,
                    "sic_codes": "[]",
                    "company_status": "not_found",
                    "company_type": "",
                    "date_of_creation": "",
                    "address_line_1": "",
                    "address_line_2": "",
                    "locality": "",
                    "region": "",
                    "postcode": "",
                    "country": "",
                    "full_address": "",
                    "channel": channel,
                })
            else:
                rows.append({
                    **result,
                    "sic_codes": json.dumps(result.get("sic_codes", [])),
                    "channel": channel,
                })

            if done % 10 == 0 or done == total:
                print(f"  Fetched {done}/{total} companies ...")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved company data ->{output_path}")
    return df
