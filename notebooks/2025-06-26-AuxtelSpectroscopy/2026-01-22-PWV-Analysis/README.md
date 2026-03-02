# README — PWV Analysis (AuxTel Spectroscopy)

- author: Sylvie Dagoret-Campagne
- creation date: 2026-02-07
- last update: 2026-03-02
- affiliation: IJCLab
- kernels: `w_2026_02` (USDF), `base` (emac), `conda_py313` (laptop)

## Overview

This notebook series analyses the **Precipitable Water Vapour (PWV)** retrieved from AuxTel hologram spectroscopy (Spectractor pipeline output). The PWV values are compared across time, per target, per filter, and against independent external references (FGCM, MERRA2). The series also studies the impact of quality selection cuts on PWV accuracy and resolution.

All notebooks rely on the shared configuration file `PWV00_parameters.py` and the `mysitcom.auxtel.pwv` module.

---

## Configuration file

- **`PWV00_parameters.py`**: defines the input data paths (parquet/npy files from Spectractor), the Butler collection version tag, the list of targets, and the cut thresholds (D_CCD, CHI2_FIT). Must be loaded with `from PWV00_parameters import *` at the start of each notebook.

---

## Notebooks

### PWV01 — PWV spectrogram and residuals per observation
**`PWV01_ExploreHoloQualityPWV-gramrumdiff.ipynb`**

Displays PWV as a function of time, in spectrogram form (PWV spectrum vs time) and as individual measurements. Also shows the difference (residual) between consecutive PWV measurements per observation. Diagnostic plots of D_CCD and CHI2_FIT quality parameters are produced for each filter (empty, OG550, BG40), with and without colour-coding by stellar spectral type (SED type). Figures saved in `figs_PWV01/`.

### PWV02 — PWV deviation with respect to nightly average
**`PWV02_ExploreHoloQualityPWV-avnightshift.ipynb`**

Studies the deviation of each PWV measurement relative to the nightly average (night shift). The normalisation and averaging are performed per target and per filter using `normalize_column_data_bytarget_byfilter` and `shiftaverage_column_data_byfilter`. This allows identifying nights with unusual PWV variability. Figures saved in `figs_PWV02/`.

### PWV03 — Cross-comparison with FGCM
**`PWV03_ComparePWV-FGCM.ipynb`**

Compares the PWV retrieved by Spectractor with the FGCM (Forward Global Calibration Method) PWV estimates. Plots PWV vs time in spectrogram and spectrum form, and computes the residuals between the two datasets per observation and per filter. Figures saved in `figs_PWV03/`.

### PWV04 — Cross-comparison with MERRA2
**`PWV04_ComparePWV-MERRA2.ipynb`**

Same comparison as PWV03 but using the MERRA2 atmospheric reanalysis model as the PWV reference. Allows assessing the consistency of the Spectractor PWV with an independent meteorological dataset.

### PWV05 — PWV comparison across all targets (no cuts)
**`PWV05_ComparePWV-alltargets.ipynb`**

Shows PWV and ΔPW V vs time and their histograms for all observed targets simultaneously. Compares CALSPEC and Gaia targets side by side. No quality selection cut is applied.

### PWV05b — PWV comparison across all targets (with selection cuts)
**`PWV05b_ComparePWV-alltargets-withSelectionCuts.ipynb`**

Same as PWV05 but applies the quality selection cuts defined in `PWV00_parameters.py`. Side-by-side comparison of PWV distributions without and with cuts. Results saved in `data_PWV05b/`, figures in `figs_PWV05b/`.

### PWV06 — PWV comparison between CALSPEC and Gaia targets (no cuts)
**`PWV06_ComparePWV-CalspecGaia.ipynb`**

Focuses on the comparison of PWV measurements between CALSPEC standard stars and Gaia targets. Plots PWV vs time and difference histograms split by target category. Figures saved in `figs_PWV06/`.

### PWV06b — PWV comparison between CALSPEC and Gaia targets (with selection cuts)
**`PWV06b_ComparePWV-CalspecGaia-withSelectionCuts.ipynb`**

Same as PWV06 with quality selection cuts applied. Allows evaluating the impact of data quality filtering on the PWV accuracy and resolution estimated separately for CALSPEC and Gaia targets. Results saved in `data_PWV06b/`, figures in `figs_PWV06b/`.

---

## Output structure

| Directory | Content |
|---|---|
| `figs_PWV01/` | Spectrogram and residual plots (PWV01) |
| `figs_PWV02/` | Nightly deviation plots (PWV02) |
| `figs_PWV03/` | FGCM comparison plots (PWV03) |
| `figs_PWV05/` | All-target PWV plots, no cuts (PWV05) |
| `figs_PWV05b/` | All-target PWV plots, with cuts (PWV05b) |
| `figs_PWV06/` | Calspec/Gaia comparison plots, no cuts (PWV06) |
| `figs_PWV06b/` | Calspec/Gaia comparison plots, with cuts (PWV06b) |
| `data_PWV05b/` | CSV/parquet output from PWV05b |
| `data_PWV06b/` | CSV/parquet output from PWV06b |

---

## Dependencies

- `mysitcom` package (install at top level: `pip install --user -e .`)
- `mysitcom.auxtel.pwv` module
- `astropy`, `pandas`, `numpy`, `matplotlib`, `seaborn`
- Input data: Spectractor output parquet files (path defined in `PWV00_parameters.py`)
