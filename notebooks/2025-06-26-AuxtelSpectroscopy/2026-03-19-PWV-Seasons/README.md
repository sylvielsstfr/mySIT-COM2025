# 2PWV Seasonal variation Analysis of AuxTel/Spectractor Data

**Author :** Sylvie Dagoret-Campagne — IJCLab/IN2P3/CNRS  
**Run version :** `run2026_v02b_cr_run2026_v02d_cr`  
**Kernel :** `conda_py313`  
**Creation date:** 2026-03-19
**Last updated :** 2026-03-30

---

## Overview

This directory contains a series of Jupyter notebooks that analyse the
**Precipitable Water Vapour (PWV)** measured by the AuxTel telescope at
Cerro Pachón using the
[Spectractor](https://github.com/LSSTDESC/Spectractor) pipeline.
The analysis covers the full multi-year dataset from 2021 to early 2026,
with a focus on:

- seasonal and annual variability of PWV,
- quality-cut selection efficiency as a function of filter and epoch,
- temporal coherence of PWV fluctuations from intra-night to multi-year timescales,
- cross-validation of AuxTel PWV against MERRA-2 reanalysis data,
- simulation of the resulting atmospheric transparency spectrum as a function of season.

The notebooks share a common configuration file (`PWV00_parameters.py`)
and rely on quality-cut JSON files stored in `data_PWV01seas/`.

---

## Directory structure

```
.
├── PWV00_parameters.py                                          # Shared configuration
├── PWV00_SpectraCountTable.ipynb                                # Spectra count & selection efficiency
│
├── PWV01seasons_allfilters-withSelectionCuts.ipynb              # PWV time series & seasonal overview
│
├── PWV02_section12_seasonal_stats.py                            # Helper script: seasonal statistics
├── PWV02seasons_SineFit_PublicationFigures.ipynb                # Sinusoidal seasonal fit
├── PWV02seasons_GPFit_PublicationFigures.ipynb                  # Gaussian Process seasonal fit
│
├── PWV03_TwoPoint_TemporalCorrelation_separateFilters.ipynb     # Intra-night DCF (binned estimator)
├── PWV03_TwoPoint_longscale.ipynb                               # Multi-day DCF (binned estimator)
├── PWV03b_TwoPoint_StructureFunction_separateFilters.ipynb      # DCF with noise-floor correction
├── PWV03c_TwoPoint_TemporalCorrelation_separateFilters_pyzdcf.ipynb  # Intra-night DCF (pyzdcf)
├── PWV03c_TwoPoint_longscale_pyzdcf.ipynb                       # Multi-day DCF (pyzdcf)
├── PWV03d_TwoPoint_TemporalCorrelation_separateFilters_sylvie.ipynb  # Intra-night DCF (Sylvie's estimator)
├── PWV03d_TwoPoint_longscale_sylvie.ipynb                       # Multi-day DCF (Sylvie's estimator)
│
├── PWV04compMerra2_allfilters-withSelectionCuts.ipynb           # AuxTel vs MERRA-2 cross-validation
│
├── PWV05simulateSeasonalTransparency_PublicationFigures.ipynb   # Seasonal transparency simulation
│
├── data_PWV01seas/          # Quality-cut JSON files and output tables
│   ├── cuts_finaldecision.json
│   ├── cuts_loose_finaldecision.json
│   └── cuts_tight_finaldecision.json
├── data_pyzdcf/             # pyzdcf intermediate output files
├── figs_PWV01seas/          # Figures from PWV01
├── figs_PWV02gp/            # Figures from PWV02 GP
├── figs_PWV02seas/          # Figures from PWV02 sine
├── figs_PWV03corr/          # Figures from PWV03 (intra-night)
├── figs_PWV03dcorr/         # Figures from PWV03b (structure function)
├── figs_PWV03longscale/     # Figures from PWV03 long-scale variants
├── figs_PWV04merra/         # Figures from PWV04
└── figs_PWV05seas/          # Figures from PWV05
```

---

## Shared configuration — `PWV00_parameters.py`

Central configuration module imported by all notebooks.  It defines:

| Symbol | Description |
|--------|-------------|
| `version_run` | Active run tag (currently `run2026_v02b_cr_run2026_v02d_cr`) |
| `extractedfilesdict` | Map from run tag to the `.npy` / `.parquet.gz` data file |
| `mergedextractedfilesdict` | Map to merged (butler + exposure list) data files |
| `PWV_FILTER_LIST` | Filters used for PWV analysis: `empty`, `OG550_65mm_1`, `FELH0600` |
| `DATETIME_COLLIMATOR` | Collimator installation date (2023-09-30); used to split epochs |
| `getSelectionCutforPWV()` | Legacy function for basic χ²/D2CCD/EXPTIME cuts |

---

## Notebooks

### `PWV00_SpectraCountTable.ipynb` — Spectra count & selection efficiency

Produces a comprehensive **selection efficiency table** for the full
AuxTel/Spectractor dataset.  For every combination of epoch (full /
post-collimator), filter subset, and cut level (none / loose / standard /
tight), the notebook computes N_total, N_selected, and
ε = N_selected / N_total (%).

**Outputs:** styled DataFrame, pivot matrix with `RdYlGn` gradient, LaTeX
`longtable` → `data_PWV01seas/table_spectra_selection_efficiency_<run>.tex`.

---

### `PWV01seasons_allfilters-withSelectionCuts.ipynb` — PWV time series & seasonal overview

Exploratory notebook providing a broad overview of the PWV dataset after
quality cuts.  It produces PWV vs. time scatter plots, nightly counts,
D2CCD / χ² diagnostics, histograms, and saves intermediate cut-flag
DataFrames to `data_PWV01seas/` for reuse by PWV02–PWV05.

**Figures saved to:** `figs_PWV01seas/`

---

### `PWV02seasons_SineFit_PublicationFigures.ipynb` — Sinusoidal seasonal fit

Fits a **pure sinusoidal model** (A sin(2π t / T + φ) + C, period free near
one year) to the full PWV time series and produces publication-quality figures:
full timeline with fit overlay, year-folded phase plot, Lomb–Scargle
periodogram, and a LaTeX seasonal statistics table (N, mean, median, std,
weighted mean and weighted std for Summer / Autumn / Winter / Spring in the
Southern-Hemisphere convention).

**Figures saved to:** `figs_PWV02seas/`

---

### `PWV02seasons_GPFit_PublicationFigures.ipynb` — Gaussian Process seasonal fit

Fits a **Gaussian Process regression** to the PWV time series using a
composite kernel (RBF + periodic + white noise), providing a non-parametric
characterisation of the annual cycle.  Produces the same set of publication
figures as the sine-fit notebook (full timeline, year-folded plot, GP
uncertainty envelope) together with a comparison of the GP posterior against
the sinusoidal fit.  In the final version the purely sinusoidal model is
preferred; the GP fit serves as a consistency check.

**Figures saved to:** `figs_PWV02gp/`

---

### `PWV02_section12_seasonal_stats.py` — Helper: seasonal statistics

Standalone Python script (not a notebook) intended to be pasted as Jupyter
cells at the end of any PWV02 notebook.  It computes per-season summary
statistics (Southern-Hemisphere convention: DJF = Summer, MAM = Autumn,
JJA = Winter, SON = Spring) and exports the result as a LaTeX table.

---

### Two-point temporal correlation — family of PWV03 notebooks

This family of seven notebooks measures the **two-point temporal correlation
function (DCF)** of PWV fluctuations at Cerro Pachón.  Three independent
estimators are implemented, each in an *intra-night* variant (lags up to ~10 h,
with per-night mean subtraction) and a *long-timescale* variant (lags from 1 min
to ~500 days, without mean subtraction):

| Notebook | Timescale | Estimator |
|----------|-----------|-----------|
| `PWV03_TwoPoint_TemporalCorrelation_separateFilters` | intra-night | direct binned pairs |
| `PWV03_TwoPoint_longscale` | multi-day | direct binned pairs |
| `PWV03b_TwoPoint_StructureFunction_separateFilters` | intra-night | binned + noise-floor correction |
| `PWV03c_TwoPoint_TemporalCorrelation_separateFilters_pyzdcf` | intra-night | pyzdcf library |
| `PWV03c_TwoPoint_longscale_pyzdcf` | multi-day | pyzdcf library |
| `PWV03d_TwoPoint_TemporalCorrelation_separateFilters_sylvie` | intra-night | Sylvie's pair-based DCF |
| `PWV03d_TwoPoint_longscale_sylvie` | multi-day | Sylvie's pair-based DCF |

#### Common methodology

All notebooks compute C(Δt) = ⟨δPWV(t) δPWV(t+Δt)⟩ / σ², where δPWV is the
mean-subtracted PWV and σ² is computed at zero lag including the instrumental
repeatability term σ_rep ≈ 0.12–0.15 mm.  Pairs are binned in
logarithmically-spaced lag intervals and results are fitted with an exponential
model C(Δt) = A exp(−Δt/τ) to extract the decorrelation timescale τ.

#### Key variants

- **PWV03** (direct binned): reference implementation; C(0) < 1 because
  instrumental noise is not accounted for in the denominator.
- **PWV03b** (noise-floor correction): fixes C(Δt→0) → 1 by subtracting
  σ²_stat + σ²_rep from the zero-lag variance before normalising.
- **PWV03c** (pyzdcf): uses the `pyzdcf` library (Ziv Ben-Yaacov Discrete
  Correlation Function), which handles uneven sampling and returns bootstrap
  error bars on C(Δt).
- **PWV03d** (Sylvie's estimator): custom pair-based DCF originally developed
  in the OLD_ exploratory notebooks; provides an independent cross-check.

**Figures saved to:** `figs_PWV03corr/`, `figs_PWV03dcorr/`, `figs_PWV03longscale/`

---

### `PWV04compMerra2_allfilters-withSelectionCuts.ipynb` — AuxTel vs MERRA-2 cross-validation

Cross-validates AuxTel/Spectractor PWV against **MERRA-2** reanalysis
(TQV field, 1-hour cadence at the Cerro Pachón grid point) via
`pandas.merge_asof`.  Produces scatter diagrams, Pearson/Spearman
correlations, linear regression fits, and per-season (DJF/JJA) ellipses.
Optionally restricted to the post-collimator epoch.

**Figures saved to:** `figs_PWV04merra/`

---

### `PWV05simulateSeasonalTransparency_PublicationFigures.ipynb` — Seasonal transparency simulation

Uses the **`getObsAtmo` atmospheric emulator** (ObsAtmo class, LSST site) to
simulate the full atmospheric transparency spectrum T(λ) at zenith (airmass = 1)
for each observation in the dataset, injecting the measured PWV, ozone column,
and VAOD parameters.  Restricted to the `empty` filter (white-light spectra) and
clear-sky observations (VAOD < 0.1, ozone > 20 db).

The notebook produces three families of publication-quality figures:

1. **Raw seasonal overlay** — all T(λ) curves colour-coded by season.
2. **Seasonal envelopes** — pixel-wise median ± 16th–84th percentile band per
   season (one panel per season in a 2×2 grid).
3. **Density-coloured bundles (2×2 grid)** — each curve is coloured by its
   intra-season PWV percentile rank using the season's sequential colormap
   (Reds / Oranges / Blues / Greens); the most saturated colour marks the
   median-PWV curves; a KDE density field is rendered as a grey background.
4. **GridSpec 5×1 summary figure** — top panel: full-year distribution of
   T(λ) coloured by annual PWV percentile (YlOrRd palette) with the annual
   median in dashed black.  Four lower panels: ratio T_season / T_annual_median
   for each season (same per-season colormaps as in the 2×2 figure), with a
   dashed reference line at ratio = 1.  Saved as
   `figs_PWV05seas/PWV05_annual_and_seasonal_ratios.pdf`.

**Figures saved to:** `figs_PWV05seas/`

---

## Quality cuts

Three levels of quality cuts are available, stored as JSON files in
`data_PWV01seas/`.  All three apply per-filter thresholds on the same set
of Spectractor parameters:

| Parameter | Description |
|-----------|-------------|
| `CHI2_FIT_norm`, `chi2_ram_norm`, `chi2_rum_norm` | Normalised χ² of the spectral fit |
| `D2CCD`, `D_CCD [mm]_ram`, `D_CCD [mm]_rum` | Distance from hologram to CCD [mm] |
| `MEANFWHM` | Mean PSF FWHM [pixels] |
| `PIXSHIFT`, `shift_x`, `shift_y` | Pixel-level position offsets |
| `TRACE_R` | Trace radius [pixels] |
| `alpha_0_1`, `alpha_0_2`, `alpha_1_1` | PSF Moffat shape parameters |
| `alpha_pix [pix]` | Chromatic PSF shift |
| `angle [deg]` | Dispersion angle |
| `gamma_*` | PSF wing parameters |
| `reso [nm]` | Spectral resolution |
| `PSF_REG` | PSF regularisation |
| `P [hPa]` | Atmospheric pressure |

| Cut file | Strictness | Typical usage |
|----------|------------|---------------|
| `cuts_loose_finaldecision.json` | Loose | PWV05 transparency simulation |
| `cuts_finaldecision.json` | Standard | general exploration |
| `cuts_tight_finaldecision.json` | Tight | publication figures |

The `FLAG_LOOSE_CUTS` / `FLAG_TIGHT_CUTS` boolean flags at the top of each
notebook select which cut file is loaded.

---

## Instrumental repeatability

An important constant used throughout the correlation notebooks is the
**instrumental repeatability** of the PWV measurement:

> σ_rep ≈ 0.12 – 0.15 mm

This term arises from systematic effects in Spectractor (PSF model
imperfections, flat-field residuals, etc.) that are not captured by the
per-exposure statistical error `PWV_err`.  It must be included in the
noise-floor of the correlation function normalisation to obtain C(Δt→0) = 1.

---

## Southern-Hemisphere season convention

All notebooks use the **Southern-Hemisphere** astronomical season definition
appropriate for Cerro Pachón (Chile):

| Season | Months | Typical PWV |
|--------|--------|-------------|
| Summer (DJF) | December, January, February | high |
| Autumn (MAM) | March, April, May | intermediate |
| Winter (JJA) | June, July, August | low (dry) |
| Spring (SON) | September, October, November | intermediate |

The mapping is encoded in the `SEASON_MAP` dictionary and `SEASON_ORDER` /
`SEASON_COLORS` lists defined in each notebook (and reproduced in the shared
parameters file).

---

## Dependencies

- Python ≥ 3.10 (tested with 3.13.3)
- `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`
- `astropy`
- `sklearn` (for `NearestNeighbors`, `KDTree`)
- `pyzdcf` (for PWV03c notebooks)
- `getObsAtmo` (for PWV05 transparency simulation)
- `mysitcom` (local package — install with `pip install --user -e .` at the repo root)
- `getCalspec` (local or installed package)

---

## Data files

Input data files are **not stored in this directory**.  They are referenced
via relative paths defined in `PWV00_parameters.py`:

- Spectractor outputs (`.npy` or `.parquet.gz`) in
  `../2025-06-26-SpectractorExtraction-FromButler/data/spectro_from_corentin/`
- Merged files (Spectractor + exposure list) in
  `../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/`
- MERRA-2 CSV in
  `../2025-09-16-SpectroMerra2/MerradataMerged/`
