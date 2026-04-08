"""Extract and analyse company name keywords from the data itself.

Keywords are NOT hardcoded -they are discovered by analysing which words
appear frequently in each channel's company names and are statistically
discriminative (using chi-squared scores).
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Common stop words to ignore when tokenising company names
_STOP_WORDS = {
    "ltd", "limited", "plc", "inc", "llp", "uk", "group", "holdings",
    "the", "and", "of", "for", "in", "a", "an", "co", "company",
}


def tokenize_name(name: str) -> list[str]:
    """Lowercase, strip punctuation, and split a company name into tokens."""
    name = re.sub(r"[^a-zA-Z0-9\s]", " ", name.lower())
    return [t for t in name.split() if t and t not in _STOP_WORDS and len(t) > 1]


# ------------------------------------------------------------------
# Keyword discovery from data
# ------------------------------------------------------------------

def discover_keywords(
    df: pd.DataFrame,
    name_col: str = "matched_name",
    min_occurrences: int = 2,
    top_n: int = 30,
) -> dict:
    """Analyse company names per channel and discover discriminative keywords.

    Returns
    -------
    dict with keys:
        - channel_word_counts: {channel: Counter}
        - keyword_scores: {channel: [(word, chi2_score), ...]}
        - all_keywords: sorted list of all selected keywords across channels
    """
    names = df[name_col].fillna(df["input_name"]).fillna("")

    # Tokenise all names
    channel_tokens: dict[str, list[str]] = {}
    for channel in df["channel"].unique():
        mask = df["channel"] == channel
        tokens: list[str] = []
        for name in names[mask]:
            tokens.extend(tokenize_name(str(name)))
        channel_tokens[channel] = tokens

    channel_word_counts = {ch: Counter(toks) for ch, toks in channel_tokens.items()}

    # Build a word x channel count matrix for chi-squared scoring
    all_words = set()
    for counts in channel_word_counts.values():
        all_words |= {w for w, c in counts.items() if c >= min_occurrences}

    if not all_words:
        return {
            "channel_word_counts": channel_word_counts,
            "keyword_scores": {},
            "all_keywords": [],
        }

    channels = sorted(channel_word_counts.keys())
    word_list = sorted(all_words)

    # Observed counts matrix: rows=words, cols=channels
    observed = np.array([
        [channel_word_counts[ch].get(w, 0) for ch in channels]
        for w in word_list
    ], dtype=float)

    # Chi-squared: compare observed vs expected (row_total * col_total / grand_total)
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    grand_total = observed.sum()

    expected = (row_totals * col_totals) / grand_total
    # Avoid division by zero
    expected = np.where(expected == 0, 1e-10, expected)
    chi2_contributions = (observed - expected) ** 2 / expected

    # Per channel: score each word by how much it over-represents that channel
    keyword_scores: dict[str, list[tuple[str, float]]] = {}
    for j, ch in enumerate(channels):
        # Only consider words that are over-represented (observed > expected)
        scores = []
        for i, w in enumerate(word_list):
            if observed[i, j] > expected[i, j]:
                scores.append((w, float(chi2_contributions[i, j])))
        scores.sort(key=lambda x: x[1], reverse=True)
        keyword_scores[ch] = scores[:top_n]

    # Collect all selected keywords
    all_keywords = sorted({w for scored in keyword_scores.values() for w, _ in scored})

    return {
        "channel_word_counts": channel_word_counts,
        "keyword_scores": keyword_scores,
        "all_keywords": all_keywords,
    }


# ------------------------------------------------------------------
# Feature building using discovered keywords
# ------------------------------------------------------------------

def build_keyword_features(
    df: pd.DataFrame,
    keywords: list[str],
    name_col: str = "matched_name",
) -> pd.DataFrame:
    """One-hot encode discovered keyword presence for each company.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *name_col* and ``input_name``.
    keywords : list[str]
        The data-driven keywords discovered by ``discover_keywords()``.

    Returns a DataFrame with one boolean column per keyword, same index as *df*.
    """
    names = df[name_col].fillna(df["input_name"]).fillna("")

    rows: list[dict[str, int]] = []
    for name in names:
        tokens = set(tokenize_name(str(name)))
        rows.append({f"kw_{kw}": int(kw in tokens) for kw in keywords})

    return pd.DataFrame(rows, index=df.index)


def keyword_channel_scores(
    name: str,
    keyword_scores: dict[str, list[tuple[str, float]]],
) -> dict[str, float]:
    """Score a company name against each channel using discovered keyword weights.

    Returns {channel: normalised_score}.
    """
    tokens = set(tokenize_name(name))
    scores: dict[str, float] = {}

    for channel, scored_words in keyword_scores.items():
        if not scored_words:
            scores[channel] = 0.0
            continue
        max_possible = sum(s for _, s in scored_words)
        hit_score = sum(s for w, s in scored_words if w in tokens)
        scores[channel] = hit_score / max_possible if max_possible > 0 else 0.0

    return scores


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------

def generate_keyword_report(
    keyword_data: dict,
    output_path: Path,
) -> str:
    """Generate a text report of discovered keywords per channel."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("COMPANY NAME KEYWORD ANALYSIS")
    lines.append("=" * 80)

    for ch, counts in keyword_data["channel_word_counts"].items():
        top = counts.most_common(20)
        lines.append(f"\n--- {ch} (top 20 most frequent words) ---")
        for word, count in top:
            lines.append(f"  {word:>20s}: {count}")

    lines.append("\n" + "=" * 80)
    lines.append("DISCRIMINATIVE KEYWORDS PER CHANNEL (chi-squared)")
    lines.append("=" * 80)

    for ch, scored in keyword_data["keyword_scores"].items():
        lines.append(f"\n--- {ch} ---")
        for word, score in scored:
            lines.append(f"  {word:>20s}: {score:.2f}")

    lines.append(f"\n  Total keywords selected: {len(keyword_data['all_keywords'])}")

    report = "\n".join(lines)
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n  Saved keyword report ->{output_path}")
    return report
