# 2026-03-19 — PWV Seasonal Analysis of AuxTel/Spectractor Data

**Author :** Sylvie Dagoret-Campagne — IJCLab/IN2P3/CNRS  
**Run version :** `run2026_v02b_cr_run2026_v02d_cr`  
**Kernel :** `conda_py313`

---

## Overview

This directory contains a series of Jupyter notebooks that analyse the
**Precipitable Water Vapour (PWV)** measured by the AuxTel telescope at
Cerro Pachon using the
[Spectractor](https://github.com/LSSTDESC/Spectractor) pipeline.  The
analysis covers the full multi-year dataset from 2021 to early 2026,
with a focus on:

- seasonal and annual variability of PWV,
- quality-cut selection efficiency as a function of filter and epoch,
- temporal coherence of PWV fluctuations on intra-night timescales,
- cross-validation of AuxTel PWV against MERRA-2 reanalysis data.

The notebooks share a common configuration file (`PWV00_parameters.py`)
and rely on quality-cut JSON files stored in `data_PWV01seas/`.

---

## Directory structure

```
.
├── PWV00_parameters.py                    # Shared configuration (run version, file paths, cut parameters)
├── PWV00_SpectraCountTable.ipynb          # Spectra count & selection efficiency table
├── PWV01seasons_allfilters-withSelectionCuts.ipynb   # PWV time series & seasonal overview
├── PWV02seasons_SineFit_PublicationFigures.ipynb     # Sinusoidal fit & publication figures
├── PWV02_section12_seasonal_stats.py      # Helper script: seasonal statistics (to be pasted into PWV02)
├── PWV03_TwoPoint_TemporalCorrelation_separateFilters.ipynb  # Two-point temporal correlation
├── PWV04compMerra2_allfilters-withSelectionCuts.ipynb        # AuxTel vs MERRA-2 cross-validation
├── data_PWV01seas/                        # Quality-cut JSON files and output tables
│   ├── cuts_finaldecision.json            # Standard quality cuts
│   ├── cuts_loose_finaldecision.json      # Loose quality cuts
│   └── cuts_tight_finaldecision.json      # Tight quality cuts
├── figs_PWV01seas/                        # Figures produced by PWV01
├── figs_PWV02seas/                        # Figures produced by PWV02
├── figs_PWV03corr/                        # Figures produced by PWV03
└── figs_PWV04merra/                       # Figures produced by PWV04
```

---

## Shared configuration — `PWV00_parameters.py`

Central configuration module imported by all notebooks.  It defines:

| Symbol | Description |
|--------|-------------|
| `version_run` | Active run tag (currently `run2026_v02b_cr_run2026_v02d_cr`) |
| `extractedfilesdict` | Map from run tag to the `.npy` / `.parquet.gz` data file |
| `mergedextractedfilesdict` | Map to merged (butler + exposure list) data files |
| `PWV_FILTER_LIST` | Filters used for PWV analysis: `empty`, `OG550_65mm_1`, `SDSSr`, `FELH0600` |
| `DATETIME_COLLIMATOR` | Collimator installation date (2023-09-30); used to split epochs |
| `getSelectionCutforPWV()` | Legacy function for basic chi²/D2CCD/EXPTIME cuts |

---

## Notebooks

### `PWV00_SpectraCountTable.ipynb` — Spectra count & selection efficiency

Produces a comprehensive **selection efficiency table** for the full
AuxTel/Spectractor dataset.  For every combination of:

- **Epoch** : full dataset vs. post-collimator only (after 2023-09-30),
- **Filter subset** : all PWV filters / `empty` only / `OG550_65mm_1` only,
- **Cut level** : no cuts / loose / standard / tight,

the notebook computes the number of spectra before cuts (*N*_total),
the number surviving the cuts (*N*_selected), and the selection efficiency
ε = *N*_selected / *N*_total (%).

**Outputs:**
- Styled pandas DataFrame with colour-coded efficiency (green / yellow / red),
- Pivot efficiency matrix with `RdYlGn` gradient,
- LaTeX `longtable` saved to `data_PWV01seas/table_spectra_selection_efficiency_<run>.tex`.

---

### `PWV01seasons_allfilters-withSelectionCuts.ipynb` — PWV time series & seasonal overview

Exploratory notebook providing a broad overview of the PWV dataset.  It loads
the Spectractor output, applies the quality cuts (selectable via flags), and
produces:

- PWV vs. time scatter plots (all filters, colour-coded),
- nightly bar charts of spectra counts,
- D2CCD and χ² diagnostic plots vs. time and by filter/target,
- histograms of PWV, diffPWV, D2CCD and χ² before and after cuts,
- comparison of PWV distributions with and without the collimator epoch split.

The notebook saves intermediate cut-flag DataFrames to `data_PWV01seas/` for
reuse by downstream notebooks (PWV02–PWV04).

**Figures saved to:** `figs_PWV01seas/`

---

### `PWV02seasons_SineFit_PublicationFigures.ipynb` — Sinusoidal fit & publication figures

Focuses on the **annual PWV cycle** at Cerro Pachon.  After applying quality
cuts (standard or tight), the notebook:

1. Fits a sinusoidal model with a free period close to one year to the PWV time
   series on the MJD axis:
   PWV(t) = A · sin(2π t / T + φ) + C.
2. Produces a **full timeline** panel (all years, with fit overlay) and a
   **year-folded (phase) panel** (Jan → Dec, one colour per year).
3. Computes a **Lomb–Scargle periodogram** to provide independent statistical
   evidence for the dominant annual period.
4. Generates a **seasonal statistics table** (Summer DJF / Autumn MAM /
   Winter JJA / Spring SON) including N, mean, median, std, weighted mean and
   weighted std for each Southern-Hemisphere season.  The table is exported as
   LaTeX.

All figures are publication-quality (serif fonts, 150 dpi, GridSpec layout)
and saved as PDF.

**Figures saved to:** `figs_PWV02seas/`

---

### `PWV02_section12_seasonal_stats.py` — Helper: seasonal statistics

Standalone Python script (not a notebook) intended to be **pasted as two
Jupyter cells** at the end of `PWV02`.  It computes per-season summary
statistics (N, mean, median, std, weighted mean and weighted std) using the
Southern-Hemisphere season convention (DJF = Summer, MAM = Autumn, JJA =
Winter, SON = Spring) and exports the result as a LaTeX table.

---

### `PWV03_TwoPoint_TemporalCorrelation_separateFilters.ipynb` — Two-point temporal correlation

Quantifies the **temporal coherence** of intra-night PWV fluctuations on
timescales from ~1 minute to ~10 hours, for each filter separately.

The analysis is based on the **two-point temporal correlation function**:

C(Δt) = ⟨δPWV(t) δPWV(t+Δt)⟩ / ⟨δPWV²⟩

where δPWV(t) is the nightly-mean-subtracted PWV.  Pairs of observations
within the same night are formed, binned in logarithmically-spaced lag bins,
and averaged.  An exponential model C(Δt) = A exp(−Δt/τ) is fitted to
extract the **decorrelation timescale τ** for each filter.

This timescale is a key input for photometric calibration strategies: it
sets the window over which a single PWV measurement can be considered
representative of the atmospheric conditions for a nearby exposure.

**Figures saved to:** `figs_PWV03corr/`

---

### `PWV04compMerra2_allfilters-withSelectionCuts.ipynb` — AuxTel vs MERRA-2 cross-validation

Cross-validates the AuxTel/Spectractor PWV measurements against the
**MERRA-2** reanalysis product (`inst1_2d_asm_Nx`, TQV field), available at
1-hour cadence at the Cerro Pachon grid point.

The notebook:

1. Matches each AuxTel observation to its nearest MERRA-2 timestamp via
   `pandas.merge_asof` (direction = nearest), and records the signed time
   offset Δt_match.
2. Computes Pearson and Spearman correlation coefficients, plots scatter
   diagrams and correlation ellipses, and fits a linear regression
   PWV_AuxTel = a · TQV_MERRA2 + b.
3. Splits the matched dataset by **Southern-Hemisphere season** (dry Winter
   JJA vs. wet Summer DJF) and compares regression slopes and correlation
   ellipses per season.
4. Optionally restricts the analysis to the post-collimator epoch
   (`FLAG_WITHCOLLIMATOR = True`).

**Figures saved to:** `figs_PWV04merra/`

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

| Cut file | Strictness | Typical efficiency |
|----------|------------|--------------------|
| `cuts_loose_finaldecision.json` | Loose | high |
| `cuts_finaldecision.json` | Standard | medium |
| `cuts_tight_finaldecision.json` | Tight | low |

---

## Dependencies

- Python ≥ 3.10 (tested with 3.13)
- `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`
- `astropy`
- `mysitcom` (local package — install with `pip install --user -e .` at the repo root)
- `getCalspec` (local or installed package)
- `sklearn` (for `NearestNeighbors`, `KDTree`)

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
