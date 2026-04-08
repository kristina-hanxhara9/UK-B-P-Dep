"""Build SIC-code-to-channel frequency tables and distribution reports."""

import json
from pathlib import Path

import pandas as pd

# Condensed UK SIC 2007 descriptions for the most common codes found in
# builders merchants, plumbing merchants, and department stores.
SIC_DESCRIPTIONS: dict[str, str] = {
    "46130": "Agents involved in the sale of timber and building materials",
    "46140": "Agents involved in the sale of machinery, industrial equipment, ships and aircraft",
    "46150": "Agents involved in the sale of furniture, household goods, hardware and ironmongery",
    "46180": "Agents specialised in the sale of other particular products",
    "46190": "Agents involved in the sale of a variety of goods",
    "46620": "Wholesale of machine tools",
    "46630": "Wholesale of mining, construction and civil engineering machinery",
    "46690": "Wholesale of other machinery and equipment",
    "46710": "Wholesale of solid, liquid and gaseous fuels and related products",
    "46720": "Wholesale of metals and metal ores",
    "46730": "Wholesale of wood, construction materials and sanitary equipment",
    "46740": "Wholesale of hardware, plumbing and heating equipment and supplies",
    "46750": "Wholesale of chemical products",
    "46760": "Wholesale of other intermediate products",
    "46900": "Non-specialised wholesale trade",
    "47110": "Retail sale in non-specialised stores with food, beverages or tobacco predominating",
    "47190": "Other retail sale in non-specialised stores",
    "47210": "Retail sale of fruit and vegetables in specialised stores",
    "47410": "Retail sale of computers, peripheral units and software in specialised stores",
    "47430": "Retail sale of audio and video equipment in specialised stores",
    "47510": "Retail sale of textiles in specialised stores",
    "47520": "Retail sale of hardware, paints and glass in specialised stores",
    "47530": "Retail sale of carpets, rugs, wall and floor coverings in specialised stores",
    "47540": "Retail sale of electrical household appliances in specialised stores",
    "47590": "Retail sale of furniture, lighting equipment and other household articles in specialised stores",
    "47710": "Retail sale of clothing in specialised stores",
    "47720": "Retail sale of footwear and leather goods in specialised stores",
    "47750": "Retail sale of cosmetic and toilet articles in specialised stores",
    "47780": "Other retail sale of new goods in specialised stores",
    "47910": "Retail sale via mail order houses or via Internet",
    "47990": "Other retail sale not in stores, stalls or markets",
    "41100": "Development of building projects",
    "41201": "Construction of commercial buildings",
    "41202": "Construction of domestic buildings",
    "42110": "Construction of roads and motorways",
    "42210": "Construction of utility projects for fluids",
    "43110": "Demolition",
    "43120": "Site preparation",
    "43210": "Electrical installation",
    "43220": "Plumbing, heat and air-conditioning installation",
    "43290": "Other construction installation",
    "43310": "Plastering",
    "43320": "Joinery installation",
    "43330": "Floor and wall covering",
    "43341": "Painting",
    "43342": "Glazing",
    "43390": "Other building completion and finishing",
    "43991": "Scaffold erection",
    "43999": "Other specialised construction activities n.e.c.",
    "68100": "Buying and selling of own real estate",
    "68209": "Other letting and operating of own or leased real estate",
    "68310": "Real estate agencies",
    "70100": "Activities of head offices",
    "70210": "Public relations and communication activities",
    "70229": "Management consultancy activities other than financial management",
    "82990": "Other business support service activities n.e.c.",
}


def build_sic_channel_matrix(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Create a SIC-code xchannel frequency matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Company data with columns ``sic_codes`` (JSON list string) and ``channel``.
    output_path : Path
        Where to save the resulting CSV.

    Returns
    -------
    pd.DataFrame
        Crosstab with SIC codes as rows and channels as columns.
    """
    # Expand the JSON-encoded sic_codes lists
    exploded = df.copy()
    exploded["sic_codes"] = exploded["sic_codes"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    exploded = exploded.explode("sic_codes").dropna(subset=["sic_codes"])

    matrix = pd.crosstab(exploded["sic_codes"], exploded["channel"])
    matrix.index.name = "sic_code"

    # Add description column
    matrix.insert(0, "description", matrix.index.map(
        lambda c: SIC_DESCRIPTIONS.get(c, "")
    ))

    matrix.to_csv(output_path)
    print(f"  Saved SIC xchannel matrix ->{output_path}")
    return matrix


def print_distribution(matrix: pd.DataFrame):
    """Pretty-print (and optionally save) the SIC distribution per channel."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("SIC CODE DISTRIBUTION BY CHANNEL")
    lines.append("=" * 80)

    # Drop the description column for numeric summaries
    numeric = matrix.drop(columns=["description"], errors="ignore")

    for channel in numeric.columns:
        col = numeric[channel]
        top = col[col > 0].sort_values(ascending=False)
        lines.append(f"\n--- {channel} ({int(top.sum())} total SIC occurrences) ---")
        for sic, count in top.items():
            desc = SIC_DESCRIPTIONS.get(str(sic), "")
            lines.append(f"  {sic}  ({int(count):>3})  {desc}")

    print("\n".join(lines))
