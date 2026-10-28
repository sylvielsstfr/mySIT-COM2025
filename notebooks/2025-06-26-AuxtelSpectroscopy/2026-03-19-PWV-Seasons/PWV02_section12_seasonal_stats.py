"""
Section §12 à insérer dans PWV02 — Seasonal statistics table.

Coller ce contenu comme deux nouvelles cellules Jupyter à la fin du notebook,
juste avant la cellule §11 (Summary), ou bien après §11.

─────────────────────────────────────────────────────────────────────────────
CELL 1  (Markdown)
─────────────────────────────────────────────────────────────────────────────
---
## 12. Seasonal PWV statistics

Seasons defined for the **Southern Hemisphere** (Cerro Pachon):
- **Summer (DJF)** : December, January, February  (months 12, 1, 2)
- **Autumn (MAM)** : March, April, May             (months 3, 4, 5)
- **Winter (JJA)** : June, July, August            (months 6, 7, 8)
- **Spring (SON)** : September, October, November  (months 9, 10, 11)

Statistics computed over all years and all PWV-sensitive filters after
tight quality cuts:
- **N**        : number of individual spectrogram measurements
- **Mean**     : arithmetic mean PWV  [mm]
- **Median**   : median PWV  [mm]
- **Std**      : standard deviation  [mm]
- **Wt. mean** : weighted mean (weights = 1/σ²)
- **Wt. std**  : weighted standard deviation  [mm]

─────────────────────────────────────────────────────────────────────────────
CELL 2  (Code)
─────────────────────────────────────────────────────────────────────────────

# ── Season definition (Southern Hemisphere) ───────────────────────────────────
SEASON_MAP = {
    12: "Summer (DJF)", 1: "Summer (DJF)", 2: "Summer (DJF)",
     3: "Autumn (MAM)", 4: "Autumn (MAM)", 5: "Autumn (MAM)",
     6: "Winter (JJA)", 7: "Winter (JJA)", 8: "Winter (JJA)",
     9: "Spring (SON)",10: "Spring (SON)",11: "Spring (SON)",
}
SEASON_ORDER = ["Summer (DJF)", "Autumn (MAM)", "Winter (JJA)", "Spring (SON)"]
SEASON_SHORT = {"Summer (DJF)": "Summer", "Autumn (MAM)": "Autumn",
                "Winter (JJA)": "Winter", "Spring (SON)": "Spring"}

df_keep["season"] = df_keep["month"].map(SEASON_MAP)


def weighted_stats(vals, errs):
    \"\"\"Return weighted mean and weighted std given values and 1-sigma errors.\"\"\"
    w    = 1.0 / (errs**2 + 1e-12)
    mu_w = np.average(vals, weights=w)
    # weighted std = sqrt( sum(w*(x-mu)^2) / sum(w) )
    var_w = np.average((vals - mu_w)**2, weights=w)
    return mu_w, np.sqrt(var_w)


rows = []
for season in SEASON_ORDER:
    sub  = df_keep[df_keep["season"] == season]
    vals = sub["PWV"].values
    errs = sub["PWV_err"].values
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
        "Season"       : season,
        "N"            : len(vals),
        "Mean [mm]"    : np.mean(vals),
        "Median [mm]"  : np.median(vals),
        "Std [mm]"     : np.std(vals, ddof=1),
        "Wt. mean [mm]": mu_w,
        "Wt. std [mm]" : std_w,
    })

df_seasons = pd.DataFrame(rows).set_index("Season")

# ── display in notebook ───────────────────────────────────────────────────────
display(
    df_seasons.style
    .format({c: "{:.3f}" for c in df_seasons.columns if c != "N"})
    .format({"N": "{:d}"})
    .set_caption(f"PWV seasonal statistics — {version_run} (tight cuts)")
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

─────────────────────────────────────────────────────────────────────────────
CELL 3  (Code)
─────────────────────────────────────────────────────────────────────────────

# ── LaTeX table ───────────────────────────────────────────────────────────────
def df_to_latex_seasonal(df, version_run):
    cols_fmt = {
        "N"            : lambda x: str(int(x)),
        "Mean [mm]"    : lambda x: f"{x:.3f}",
        "Median [mm]"  : lambda x: f"{x:.3f}",
        "Std [mm]"     : lambda x: f"{x:.3f}",
        "Wt. mean [mm]": lambda x: f"{x:.3f}",
        "Wt. std [mm]" : lambda x: f"{x:.3f}",
    }

    header_latex = {
        "N"            : r"$N$",
        "Mean [mm]"    : r"$\langle\mathrm{PWV}\rangle$ [mm]",
        "Median [mm]"  : r"$\widetilde{\mathrm{PWV}}$ [mm]",
        "Std [mm]"     : r"$\sigma$ [mm]",
        "Wt. mean [mm]": r"$\bar{w}\mathrm{PWV}$ [mm]",
        "Wt. std [mm]" : r"$\sigma_w$ [mm]",
    }

    ncols = 1 + len(df.columns)
    col_spec = "l" + "r" * len(df.columns)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Seasonal statistics of the PWV at Cerro Pach\'on measured "
        r"by AuxTel/Spectractor (tight quality cuts). "
        r"Summer/Winter refer to the Southern Hemisphere. "
        r"$\bar{w}\mathrm{PWV}$ and $\sigma_w$ are the measurement-"
        r"uncertainty-weighted mean and standard deviation.}"
    )
    lines.append(r"\label{tab:pwv_seasonal}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\hline\hline")

    # header row
    header_cells = ["Season"] + [header_latex.get(c, c) for c in df.columns]
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\hline")

    season_latex = {
        "Summer (DJF)": r"Summer (DJF)",
        "Autumn (MAM)": r"Autumn (MAM)",
        "Winter (JJA)": r"Winter (JJA)",
        "Spring (SON)": r"Spring (SON)",
    }

    for season, row in df.iterrows():
        cells = [season_latex.get(season, season)]
        for col in df.columns:
            fmt = cols_fmt.get(col, lambda x: f"{x:.3f}")
            cells.append(fmt(row[col]))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


latex_str = df_to_latex_seasonal(df_seasons, version_run)

# print for copy-paste
print(latex_str)

# save to file
latex_file = f"{pathfigs}/{prefix}_{version_run}_seasonal_stats_table.tex"
with open(latex_file, "w") as fh:
    fh.write(latex_str + "\n")
print(f"\nSaved LaTeX table to: {latex_file}")
"""
