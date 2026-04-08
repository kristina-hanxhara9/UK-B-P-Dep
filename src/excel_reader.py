"""Read company names from a multi-sheet Excel file."""

import pandas as pd
from pathlib import Path


def read_companies(
    filepath: Path,
    sheet_to_channel: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """Read an Excel file and return company names grouped by channel.

    Parameters
    ----------
    filepath : Path
        Path to the .xlsx file. Each sheet represents a channel.
    sheet_to_channel : dict, optional
        Mapping of sheet names to channel names. If None, sheet names are used
        as channel names directly.

    Returns
    -------
    dict[str, list[str]]
        Mapping of channel name to list of company names.
    """
    xls = pd.ExcelFile(filepath)
    companies_by_channel: dict[str, list[str]] = {}

    for sheet_name in xls.sheet_names:
        channel = (
            sheet_to_channel.get(sheet_name, sheet_name)
            if sheet_to_channel
            else sheet_name
        )

        df = pd.read_excel(xls, sheet_name=sheet_name)
        if df.empty or df.shape[1] == 0:
            print(f"  [WARN] Sheet '{sheet_name}' is empty -skipping.")
            continue

        # Take the first column as company names
        names = (
            df.iloc[:, 0]
            .dropna()
            .astype(str)
            .str.strip()
        )
        names = names[names != ""].tolist()

        companies_by_channel[channel] = names
        print(f"  Sheet '{sheet_name}' ->channel '{channel}': {len(names)} companies")

    return companies_by_channel
