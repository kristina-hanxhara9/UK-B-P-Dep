"""Analyse SIC code overlap and uniqueness across channels."""

from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.sic_mapping import SIC_DESCRIPTIONS


# ------------------------------------------------------------------
# Set operations
# ------------------------------------------------------------------

def compute_channel_sic_sets(matrix: pd.DataFrame) -> dict[str, set[str]]:
    """Return the set of SIC codes that appear at least once per channel."""
    numeric = matrix.drop(columns=["description"], errors="ignore")
    return {
        channel: set(numeric.index[numeric[channel] > 0])
        for channel in numeric.columns
    }


def unique_sic_codes(sic_sets: dict[str, set[str]]) -> dict[str, set[str]]:
    """SIC codes that appear in *only* one channel."""
    result: dict[str, set[str]] = {}
    channels = list(sic_sets.keys())
    for ch in channels:
        others = set().union(*(sic_sets[o] for o in channels if o != ch))
        result[ch] = sic_sets[ch] - others
    return result


def shared_sic_codes(sic_sets: dict[str, set[str]]) -> set[str]:
    """SIC codes present in ALL channels."""
    sets = list(sic_sets.values())
    if not sets:
        return set()
    return set.intersection(*sets)


def overlap_analysis(sic_sets: dict[str, set[str]]) -> dict:
    """Compute pairwise and global overlaps."""
    result: dict[str, object] = {}
    channels = list(sic_sets.keys())

    # Pairwise
    for a, b in combinations(channels, 2):
        key = f"{a} ∩ {b}"
        inter = sic_sets[a] & sic_sets[b]
        result[key] = {"count": len(inter), "codes": sorted(inter)}

    # All three
    common = shared_sic_codes(sic_sets)
    result["all_channels"] = {"count": len(common), "codes": sorted(common)}

    return result


# ------------------------------------------------------------------
# Report generation
# ------------------------------------------------------------------

def generate_analysis_report(
    matrix: pd.DataFrame,
    sic_sets: dict[str, set[str]],
    output_path: Path,
) -> str:
    """Generate a comprehensive text report of SIC code analysis."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("SIC CODE ANALYSIS REPORT")
    lines.append("=" * 80)

    # Per-channel summary
    lines.append("\n1. SIC CODES PER CHANNEL")
    lines.append("-" * 40)
    for ch, codes in sic_sets.items():
        lines.append(f"  {ch}: {len(codes)} distinct SIC codes")

    # Unique codes
    lines.append("\n2. UNIQUE SIC CODES (only in one channel)")
    lines.append("-" * 40)
    uniques = unique_sic_codes(sic_sets)
    for ch, codes in uniques.items():
        lines.append(f"\n  {ch} ({len(codes)} unique):")
        for c in sorted(codes):
            desc = SIC_DESCRIPTIONS.get(c, "")
            lines.append(f"    {c}  {desc}")

    # Shared codes
    lines.append("\n3. SHARED SIC CODES (in all channels)")
    lines.append("-" * 40)
    common = shared_sic_codes(sic_sets)
    if common:
        for c in sorted(common):
            desc = SIC_DESCRIPTIONS.get(c, "")
            lines.append(f"    {c}  {desc}")
    else:
        lines.append("    (none)")

    # Pairwise overlaps
    lines.append("\n4. PAIRWISE OVERLAPS")
    lines.append("-" * 40)
    overlaps = overlap_analysis(sic_sets)
    for key, val in overlaps.items():
        if key == "all_channels":
            continue
        lines.append(f"\n  {key} — {val['count']} codes:")
        for c in val["codes"]:
            desc = SIC_DESCRIPTIONS.get(c, "")
            lines.append(f"    {c}  {desc}")

    report = "\n".join(lines)
    output_path.write_text(report)
    print(report)
    print(f"\n  Saved analysis report → {output_path}")
    return report


# ------------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------------

def plot_sic_heatmap(matrix: pd.DataFrame, output_path: Path):
    """Save a heatmap of SIC code frequencies across channels."""
    import seaborn as sns

    numeric = matrix.drop(columns=["description"], errors="ignore")
    # Keep only codes that appear at least once
    numeric = numeric.loc[numeric.sum(axis=1) > 0]

    fig, ax = plt.subplots(figsize=(8, max(6, len(numeric) * 0.35)))
    sns.heatmap(
        numeric.astype(int),
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("SIC Code Frequency by Channel")
    ax.set_ylabel("SIC Code")
    ax.set_xlabel("Channel")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved heatmap → {output_path}")
