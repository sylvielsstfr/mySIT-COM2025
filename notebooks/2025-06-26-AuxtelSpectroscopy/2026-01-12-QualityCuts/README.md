# README — Quality Cuts (AuxTel Spectroscopy)

- author: Sylvie Dagoret-Campagne
- creation date: 2026-01-16
- last update: 2026-03-02
- affiliation: IJCLab
- kernels: `w_2026_02` / `w_2026_03` (USDF), `base` (emac), `conda_py313` (laptop)

## Overview

This notebook series defines, studies and validates **quality selection cuts** on AuxTel hologram spectra processed by the Spectractor pipeline. The two main quality discriminators are the **D_CCD** (hologram-to-CCD distance, in mm) and **CHI2_FIT** (goodness of fit of the atmospheric spectrum fit). The series progresses from initial exploration of these parameters, to a systematic study of their dependence on external conditions, to a final consolidated set of cuts and an assessment of their impact on PWV accuracy and resolution.

All notebooks load the shared configuration file `QCUT00_parameters.py` and use the `mysitcom.auxtel.qualitycuts` module.

---

## Configuration file

- **`QCUT00_parameters.py`**: defines input data paths (Spectractor parquet output), Butler collection version, cut thresholds for D_CCD and CHI2_FIT, and a `DumpConfig()` function to print the active configuration. Must be imported with `from QCUT00_parameters import *`.

---

## Notebooks

### QCUT01 — Initial exploration of hologram data quality
**`QCUT01_ExploreHoloQuality.ipynb`**

First look at the distribution of quality parameters D_CCD and CHI2_FIT over time and across all targets. Scatter plots, strip plots, and histograms of D_CCD and CHI2_FIT are produced for the three filters (empty, OG550, BG40), both globally and per target. Colour-coding by stellar spectral type (SED type, from B-V colour index). Produces a summary table of mean and sigma of D_CCD and CHI2_FIT per target/filter, with a CALSPEC flag. Figures saved in `figs_QCUT01/`.

### QCUT02 — Visual inspection of Spectractor spectra via Butler
**`QCUT02_CheckFitQualityInSpectractorResults.ipynb`**

⚠️ **Requires access to the USDF Butler repository** (`/sdf/group/rubin/repo/main`).

Retrieves individual `spectractorSpectrum` products from Butler and displays their spectrum summary and spectrogram for a selected target (e.g. HD38666 = μ Col). Allows visual diagnosis of spectra with high or low CHI2_FIT. Loops over a sample of N observations to compare good and bad quality fits side by side. Figures saved in `figs_QCUT02/`.

### QCUT03 — CHI2 normalisation and quality cut per target
**`QCUT03_QualityCut-CHI2-PerTarget.ipynb`**

Normalises the CHI2_FIT values to allow meaningful comparison across targets (which have different flux levels and numbers of fitted parameters). Studies the per-target distribution of normalised chi2 and proposes target-specific cut thresholds.

### QCUT03b — CHI2 per target (variant)
**`QCUT03b_QualityCut-CHI2-PerTarget.ipynb`**

Variant of QCUT03 exploring alternative normalisation schemes or cut strategies. Figures saved in `figs_QCUT03b/`.

### QCUT04 — Impact of telescope/instrument external conditions on quality
**`QCUT04_ExploreHoloQuality_ImpactOfExternalConditions.ipynb`**

Correlates D_CCD and CHI2_FIT with external telescope/instrument parameters (e.g. airmass, pointing, rotator angle, focus). Identifies which instrumental conditions degrade the fit quality. Figures saved in `figs_QCUT04/`.

### QCUT05 — Impact of weather conditions on quality
**`QCUT05_ExploreHoloQuality_ImpactOfWeatherConditions.ipynb`**

Same correlation study as QCUT04, but focused on weather parameters (humidity, wind speed, temperature, seeing). Helps understand which atmospheric conditions lead to poor spectral fits. Figures saved in `figs_QCUT05/`.

### QCUT06 — ConsDB exploration for a single target (HD38666)
**`QCUT06_ConstDB_target38666.ipynb`**

⚠️ **Requires access to the USDF ConsDB** (`https://usdf-rsp-dev.slac.stanford.edu/consdb/`).

Queries the Consolidated Database (ConsDB) for exposure metadata (exposure time, conditions) associated with the CALSPEC target HD38666 (μ Col). Used to cross-check Spectractor results with observatory telemetry.

### QCUT07 — ConsDB exploration for external parameters (all targets)
**`QCUT07_ConstDB_ExternalParameters.ipynb`**

⚠️ **Requires access to USDF ConsDB.**

Extends QCUT06 to query ConsDB external parameters for all targets. Provides a broader view of the correlation between observing conditions stored in ConsDB and the quality of the hologram fits.

### QCUT08 — Full parameter histograms and PWV accuracy without cuts
**`QCUT08_ExploreHoloQuality_ImpactOfAllQualityCuts.ipynb`**

Produces comprehensive histograms and 2D scatter plots of all quality parameters (D_CCD, CHI2_FIT, airmass, PWV, ΔPW V…) with **no cut applied**. Computes and writes to CSV the **PWV accuracy and resolution** in the no-cut case, serving as the baseline reference. Figures saved in `figs_QCUT08/`, data saved in `data_QCUT08/`.

### QCUT09 — Parameter histograms per target with extended ranges
**`QCUT09_ExploreHoloQuality_ImpactOfAllQualityCuts_bytarget.ipynb`**

Same as QCUT08 but broken down per target, with extended parameter ranges for better visualisation. Figures saved in `figs_QCUT09/`.

### QCUT09b — Parameter histograms per target with reduced cut set
**`QCUT09b_ExploreHoloQuality_ImpactOfLessQualityCuts_bytarget.ipynb`**

Variant of QCUT09 testing a reduced (looser) set of quality cuts. Figures saved in `figs_QCUT09b/`.

### QCUT09c — Parameter histograms for Gaia targets only
**`QCUT09c_ExploreHoloQuality_ImpactOfLessQualityCuts_byGaiatarget.ipynb`**

Same exploration as QCUT09b restricted to Gaia targets. Useful to separately assess the quality of Gaia observations. Figures saved in `figs_QCUT09c/`.

### QCUT10 — Generate default cut configuration JSON file
**`QCUT10_SelectQualityCuts-GenerateDefaultConfig.ipynb`**

Studies the fraction of selected observations as a function of cut thresholds. Generates a default JSON configuration file (`cuts_default.json`) encoding the chosen cut values for all parameters. Also produces the demo cut files in `demo_cuts/`. Figures saved in `figs_QCUT10/`, data in `data_QCUT10/`.

### QCUT11 — Apply per-user cuts and compute PWV accuracy/resolution
**`QCUT11_SelectQualityCuts-CompareUsersConfig.ipynb`**

Reads JSON cut configuration files from multiple users, applies each user's cuts to the dataset, and computes the resulting PWV accuracy and resolution per filter. Writes per-user CSV result tables. Figures saved in `figs_QCUT11/`, data in `data_QCUT11/`.

### QCUT12 — Summary table: compare no-cut vs user cuts on PWV performance
**`QCUT12_SummaryQualityCuts-ComparePWV_accuracy_resolution_Users.ipynb`**

Reads the CSV files produced by QCUT08 (no-cut baseline) and QCUT11 (per-user cuts) and assembles a combined comparison table showing PWV accuracy and resolution for each user's cut configuration. Output HTML and PDF tables are written. Figures saved in `figs_QCUT12/`, data in `data_QCUT12/`.

### QCUT13 — Final decision on cut parameter configuration
**`QCUT13_SelectQualityCuts-ParameterCutConfig-FinalDecision.ipynb`**

Makes the final selection of cut thresholds based on the results of QCUT10–QCUT12. Produces the definitive cut configuration and documents the rationale. Figures saved in `figs_QCUT13/`, data in `data_QCUT13/`.

### QCUT13b — Final decision (variant)
**`QCUT13b_SelectQualityCuts-ParameterCutConfig-FinalDecision.ipynb`**

Variant of QCUT13 exploring an alternative final cut configuration. Figures saved in `figs_QCUT13b/`, data in `data_QCUT13b/`.

### QCUT14 — Performance summary for the final cut configuration
**`QCUT14_SelectQualityCuts-ParameterCutConfig-Performances.ipynb`**

Evaluates the performance (PWV accuracy, resolution, efficiency) of the final cut configuration decided in QCUT13. Produces the main performance summary figures and tables for the paper/report. Figures saved in `figs_QCUT14/`, data in `data_QCUT14/`.

### QCUT15 — Full summary: all cuts, all targets
**`QCUT15_SummaryQualityCuts-ComparePWV_accuracy_resolution_allcuts_alltargets.ipynb`**

Final consolidated comparison of PWV accuracy and resolution across all cut configurations and all targets. Extends QCUT12 to include the final cut decisions from QCUT13/14. Generates the definitive summary tables. Figures saved in `figs_QCUT15/`, data in `data_QCUT15/`.

---

## Workflow summary

```
QCUT00_parameters.py   ← shared configuration (paths, thresholds, Butler collection)
        │
        ├── QCUT01  Explore D_CCD & CHI2 distributions (no cut)
        ├── QCUT02  Visual check via Butler spectra           ← USDF only
        ├── QCUT03/03b  Chi2 normalisation per target
        ├── QCUT04  Correlate quality vs telescope params
        ├── QCUT05  Correlate quality vs weather params
        ├── QCUT06/07   ConsDB telemetry cross-check         ← USDF only
        ├── QCUT08  Baseline histograms + PWV metrics (no cut) → CSV
        ├── QCUT09/09b/09c  Per-target histograms
        ├── QCUT10  Generate default JSON cut config → cuts_default.json
        ├── QCUT11  Apply per-user cuts → per-user CSV
        ├── QCUT12  Compare no-cut vs user cuts → summary table
        ├── QCUT13/13b  Final cut decision
        ├── QCUT14  Performance of final cuts → summary figures
        └── QCUT15  Full summary: all cuts × all targets
```

---

## Output structure

| Directory / File | Content |
|---|---|
| `cuts_default.json` | Default cut configuration generated by QCUT10 |
| `demo_cuts/` | Example cut configuration files |
| `data_QCUT08/` | Baseline PWV metrics (no cut) |
| `data_QCUT10/` | Selection fraction data |
| `data_QCUT11/` | Per-user cut results |
| `data_QCUT12/` | Comparison tables no-cut vs user cuts |
| `data_QCUT13/`, `data_QCUT13b/` | Final cut data |
| `data_QCUT14/` | Final cut performance data |
| `data_QCUT15/` | Full summary data |
| `figs_QCUT*/` | Figures for each notebook |
| `table_output.html/pdf` | Final output tables |

---

## Dependencies

- `mysitcom` package (install at top level: `pip install --user -e .`)
- `mysitcom.auxtel.qualitycuts` module
- `astropy`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`
- `getCalspec` package (to flag CALSPEC targets)
- Butler access (for QCUT02): USDF kernel `w_2026_02`
- ConsDB access (for QCUT06/07): USDF kernel `w_2026_03`
- Input data: Spectractor parquet output (path defined in `QCUT00_parameters.py`)
