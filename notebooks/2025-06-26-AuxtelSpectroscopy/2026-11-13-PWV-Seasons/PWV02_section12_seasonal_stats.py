"""
Seasonal PWV statistics (section 12 of the PWV02seasons notebooks) as an importable module.

Author : Sylvie Dagoret-Campagne, IJCLab/IN2P3/CNRS
Created : 2026-03 (as a snippet to paste in the PWV02 notebooks)
Last update : 2026-11-13 : converted into a module, the type of quality cuts is no more hard-coded.

The same section is already included (paste of this snippet) in
PWV02seasons_SineFit_PublicationFigures.ipynb and PWV02seasons_GPFit_PublicationFigures.ipynb.
Use this module to get the same table in any other notebook :

    from PWV02_section12_seasonal_stats import (
        add_season_column, seasonal_stats, style_seasonal_stats, df_to_latex_seasonal, save_latex_table)

    df_keep = add_season_column(df_keep)                 # needs the column "month" (1-12)
    df_seasons = seasonal_stats(df_keep)                 # needs the columns "PWV" and "PWV_err"
    display(style_seasonal_stats(df_seasons, version_run, cutstype_name))
    latex_str = df_to_latex_seasonal(df_seasons, cutstype_name)
    save_latex_table(latex_str, f"{pathfigs}/{prefix}_{version_run}_seasonal_stats_table.tex")

Seasons are defined for the Southern Hemisphere (Cerro Pachon) :
    Summer (DJF) : December, January, February   (months 12, 1, 2)
    Autumn (MAM) : March, April, May             (months 3, 4, 5)
    Winter (JJA) : June, July, August            (months 6, 7, 8)
    Spring (SON) : September, October, November  (months 9, 10, 11)

Statistics computed over all years and all filters of the input table :
    N          : number of individual measurements
    Mean       : arithmetic mean PWV [mm]
    Median     : median PWV [mm]
    Std        : standard deviation [mm]
    Wt. mean   : weighted mean (weights = 1 / sigma^2) [mm]
    Wt. std    : weighted standard deviation [mm]
"""

import numpy as np
import pandas as pd

# ── Season definition (Southern Hemisphere) ───────────────────────────────────
SEASON_MAP = {
    12: "Summer (DJF)", 1: "Summer (DJF)", 2: "Summer (DJF)",
     3: "Autumn (MAM)", 4: "Autumn (MAM)", 5: "Autumn (MAM)",
     6: "Winter (JJA)", 7: "Winter (JJA)", 8: "Winter (JJA)",
     9: "Spring (SON)", 10: "Spring (SON)", 11: "Spring (SON)",
}
SEASON_ORDER = ["Summer (DJF)", "Autumn (MAM)", "Winter (JJA)", "Spring (SON)"]
SEASON_SHORT = {"Summer (DJF)": "Summer", "Autumn (MAM)": "Autumn",
                "Winter (JJA)": "Winter", "Spring (SON)": "Spring"}


def add_season_column(df, month_col="month", season_col="season"):
    """Add a season column computed from the month (1-12). Returns the same DataFrame."""
    df[season_col] = df[month_col].map(SEASON_MAP)
    return df


def weighted_stats(vals, errs):
    """Return weighted mean and weighted std given values and 1-sigma errors."""
    w = 1.0 / (errs**2 + 1e-12)
    mu_w = np.average(vals, weights=w)
    # weighted std = sqrt( sum(w*(x-mu)^2) / sum(w) )
    var_w = np.average((vals - mu_w)**2, weights=w)
    return mu_w, np.sqrt(var_w)


def seasonal_stats(df, pwv_col="PWV", err_col="PWV_err", season_col="season"):
    """Statistics of the PWV per season : DataFrame indexed by season, in the order of SEASON_ORDER.
    A season without data has N = 0 and NaN statistics."""
    rows = []
    for season in SEASON_ORDER:
        sub = df[df[season_col] == season]
        vals = sub[pwv_col].values
        errs = sub[err_col].values
        mask = np.isfinite(vals) & np.isfinite(errs) & (errs > 0)
        vals, errs = vals[mask], errs[mask]
        if len(vals) == 0:
            rows.append({"Season": season, "N": 0,
                         "Mean [mm]": np.nan, "Median [mm]": np.nan,
                         "Std [mm]": np.nan,
                         "Wt. mean [mm]": np.nan, "Wt. std [mm]": np.nan})
            continue
        mu_w, std_w = weighted_stats(vals, errs)
        rows.append({
            "Season": season,
            "N": len(vals),
            "Mean [mm]": np.mean(vals),
            "Median [mm]": np.median(vals),
            "Std [mm]": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
            "Wt. mean [mm]": mu_w,
            "Wt. std [mm]": std_w,
        })
    return pd.DataFrame(rows).set_index("Season")


def style_seasonal_stats(df_seasons, version_run, cutstype_name="quality cuts"):
    """Styled version of the table for the display in a notebook : display(style_seasonal_stats(...))."""
    return (
        df_seasons.style
        .format({c: "{:.3f}" for c in df_seasons.columns if c != "N"})
        .format({"N": "{:d}"})
        .set_caption(f"PWV seasonal statistics — {version_run} ({cutstype_name})")
        .set_table_styles([
            {"selector": "caption",
             "props": [("font-weight", "bold"), ("font-size", "13px"),
                       ("text-align", "left"), ("padding-bottom", "6px")]},
            {"selector": "th",
             "props": [("background-color", "#f0f0f0"), ("font-weight", "bold")]},
            {"selector": "tr:nth-child(even)",
             "props": [("background-color", "#fafafa")]},
        ])
    )


def df_to_latex_seasonal(df, cutstype_name="quality cuts", label="tab:pwv_seasonal"):
    """LaTeX table of the seasonal statistics."""
    cols_fmt = {
        "N": lambda x: str(int(x)),
        "Mean [mm]": lambda x: f"{x:.3f}",
        "Median [mm]": lambda x: f"{x:.3f}",
        "Std [mm]": lambda x: f"{x:.3f}",
        "Wt. mean [mm]": lambda x: f"{x:.3f}",
        "Wt. std [mm]": lambda x: f"{x:.3f}",
    }
    header_latex = {
        "N": r"$N$",
        "Mean [mm]": r"$\langle\mathrm{PWV}\rangle$ [mm]",
        "Median [mm]": r"$\widetilde{\mathrm{PWV}}$ [mm]",
        "Std [mm]": r"$\sigma$ [mm]",
        "Wt. mean [mm]": r"$\bar{w}\mathrm{PWV}$ [mm]",
        "Wt. std [mm]": r"$\sigma_w$ [mm]",
    }
    col_spec = "l" + "r" * len(df.columns)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Seasonal statistics of the PWV at Cerro Pach\'on measured "
        r"by AuxTel/Spectractor (" + cutstype_name.replace("-", " ") + r"). "
        r"Summer/Winter refer to the Southern Hemisphere. "
        r"$\bar{w}\mathrm{PWV}$ and $\sigma_w$ are the measurement-"
        r"uncertainty-weighted mean and standard deviation.}"
    )
    lines.append(r"\label{" + label + r"}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\hline\hline")

    header_cells = ["Season"] + [header_latex.get(c, c) for c in df.columns]
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\hline")

    for season, row in df.iterrows():
        cells = [season]
        for col in df.columns:
            fmt = cols_fmt.get(col, lambda x: f"{x:.3f}")
            cells.append(fmt(row[col]))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def save_latex_table(latex_str, latex_file):
    """Print the LaTeX table (for copy-paste) and save it in a file."""
    print(latex_str)
    with open(latex_file, "w") as fh:
        fh.write(latex_str + "\n")
    print(f"\nSaved LaTeX table to: {latex_file}")
