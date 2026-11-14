# File to centralize config parameters, path and selection files and selection functions
# for the PWV seasons notebooks (PWV01seasons, PWV02seasons, PWV03, PWV04, PWV05)
#
# author : Sylvie Dagoret-Campagne
# affiliation : IJCLab/IN2P3/CNRS
# Creation date : 2026-01-22
# last update : 2026-03-15 : Corentin files concatenated
# last update : 2026-11-13 : adapted to the merged files produced in
#                            ../2026-11-13-SpectractorExtraction-FromButler (runs 2023-2026 from Corentin Ravoux)
#
# Main changes with respect to 2026-03-19-PWV-Seasons/PWV00_parameters.py :
#   - input files are the 3 merged files of 2026-11-13-SpectractorExtraction-FromButler
#   - butler repo / collection bookkeeping removed (not needed for the analysis)
#   - Merra2 files extended up to 2026-10
#   - the columns of the two fit methods are given by PWV_COL_RAM / PWV_COL_RUM, ... :
#       "ram" : parameters fitted on the spectrogram (2D CCD image) in Spectractor
#       "rum" : parameters fitted on the 1D spectrum (projection on the dispersion axis) in Spectractor
#   - EarthLocation falls back to hard-coded coordinates if the astropy site registry is not reachable

from datetime import datetime

import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun


# ---------------------------------------------------------------------------
# Input file configuration
# ---------------------------------------------------------------------------

# Folder where the extraction + merge notebooks write their results
EXTRACTION_DIR = "../2026-11-13-SpectractorExtraction-FromButler"
MERGED_DIR = f"{EXTRACTION_DIR}/data/spectro_merged"

# Select the run version tag
#version_run = "run2026_v02b_cr_run2026_v02d_cr"   # Feb 2026 reprocessing (v02b + v02d)
#version_run = "run2026_JanToJul_cr"                # monthly runs Jan-Jul 2026
version_run = "run2023-2026_JanToJul_cr"            # all data 2023-2026 (v02b + v02d + monthly Jan-Jul 2026)

# Label used in the plots to document the Spectractor run
legendtag = {
    "run2026_v02b_cr_run2026_v02d_cr": "concat(cr_feb26_gaiaspec_calspecgaiatarget_calspecthroughput_preoct23,cr_feb26_gaiaspec_gaiatarget_calspecthroughput)",
    "run2026_JanToJul_cr": "cr_runs2026-Jan-Jul",
    "run2023-2026_JanToJul_cr": "cr_run2023-2026_JanToJul",
}

# Path of the merged files (Spectractor parameters merged with the exposure list from the butler registry)
mergedextractedfilesdict = {
    "run2026_v02b_cr_run2026_v02d_cr": f"{MERGED_DIR}/auxtel_atmosphere_feb26_gaiaspec_run2026_v02b_cr_run2026_v02d_cr_merged.npy",
    "run2026_JanToJul_cr": f"{MERGED_DIR}/auxtel_atmosphere_calspecgaiatarget_run2026_JanToJul_cr_merged.npy",
    "run2023-2026_JanToJul_cr": f"{MERGED_DIR}/auxtel_atmosphere_calspecgaiatarget_run2023-2026_JanToJul_cr_merged.npy",
}

atmfilename = mergedextractedfilesdict[version_run]
tag = legendtag[version_run]


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

FLAG_PWVFILTERS = True
# Filters used for PWV (the 2026 merged files contain empty, BG40_65mm_1/BG40, OG550_65mm_1, FELH0600)
PWV_FILTER_LIST = ["empty", "OG550_65mm_1", "SDSSr", "FELH0600"]
PWV_FILTEROG550_LIST = ["OG550_65mm_1"]


# ---------------------------------------------------------------------------
# Column names of the fit results in the merged files
# ---------------------------------------------------------------------------
PWV_COL_RAM = "PWV [mm]_ram"
PWV_ERR_COL_RAM = "PWV [mm]_err_ram"
PWV_COL_RUM = "PWV [mm]_rum"
PWV_ERR_COL_RUM = "PWV [mm]_err_rum"

OZONE_COL_RAM = "ozone [db]_ram"
OZONE_ERR_COL_RAM = "ozone [db]_err_ram"
OZONE_COL_RUM = "ozone [db]_rum"
OZONE_ERR_COL_RUM = "ozone [db]_err_rum"

# Other columns used by the cuts
CHI2_COL = "CHI2_FIT"
D2CCD_COL = "D2CCD"
EXPTIME_COL = "EXPTIME"
TARGET_COL = "TARGET"
FILTER_COL = "FILTER"
TIME_COL = "DATE-OBS"


def DumpConfig():
    """
    Dump the configuration chosen here
    """
    print("**************************************************************************")
    print("*                         DumpConfig()                                   *")
    print(f" \t - version_run = {version_run}")
    print(f" \t - annotation tag = {tag}")
    print(f" \t - merged file = {atmfilename}")
    print(f" \t - PWV filters = {PWV_FILTER_LIST}")
    print("**************************************************************************")


# ---------------------------------------------------------------------------
# Selection parameters for cuts
# ---------------------------------------------------------------------------
DCCDMINFIG = 185.0
DCCDMAXFIG = 190.0
DCCDMINCUT = 186.7
DCCDMAXCUT = 188.0
#CHI2CUT = 350.
CHI2CUT = 50.
EXPTIMECUT = 20.0
PWVMINCUT = 0.
PWVMAXCUT = 20.
OZMINCUT = 0.
OZMAXCUT = 650.0


def getSelectionCutforPWV(df,
                          pwv_ram_col=PWV_COL_RAM,
                          pwv_rum_col=PWV_COL_RUM,
                          chi2_col=CHI2_COL,
                          d2ccd_col=D2CCD_COL,
                          exptime_col=EXPTIME_COL,
                          chi2max=CHI2CUT,
                          pwvmin=PWVMINCUT,
                          pwvmax=PWVMAXCUT):
    """Standard quality selection requiring valid PWV for both methods. Returns the selected DataFrame."""
    cut = ((df[chi2_col] < chi2max)
           & (df[pwv_ram_col] > pwvmin) & (df[pwv_ram_col] < pwvmax)
           & (df[pwv_rum_col] > pwvmin) & (df[pwv_rum_col] < pwvmax)
           & (df[d2ccd_col] > DCCDMINCUT) & (df[d2ccd_col] < DCCDMAXCUT)
           & (df[exptime_col] > EXPTIMECUT))
    return df[cut]


# ---------------------------------------------------------------------------
# Photometric repeatability
# ---------------------------------------------------------------------------
FLAG_CORRECTFOR_PWV_REPEAT = True
FLAG_CORRECTFOR_PWV_REPEAT_RATIO = False

#SIGMA_PWV_REPEAT = 0.25
SIGMA_PWV_REPEAT = 0.10      # TO BE RE-ESTIMATED with the 2026 data (see PWV02 repeatability notebooks)
FACTORERR_PWV_REPEAT = 10.


# ---------------------------------------------------------------------------
# FGCM file
# ---------------------------------------------------------------------------
filename_fgcm = "../../2025-12-04-FGCM/data_FGCM04_ReadMergeAtmParams/fgcm_rdp2_prep_cLSSTCam_runs_DRP_20250417-20251214_DM-53545andDM-53697.csv"


# ---------------------------------------------------------------------------
# Merra2 (hourly data 2021 -> 2026-10-01)
# ---------------------------------------------------------------------------
MERRA2_DIR = "../2025-09-16-SpectroMerra2/MerradataMerged"
filename_m2 = f"{MERRA2_DIR}/Merge_inst1_2d_asm_Nx_M2I1NXASM-2021-2026.csv"       # meteorology (PWV, pressure, ...)
filename_m2_aer = f"{MERRA2_DIR}/Merge_tavg1_2d_aer_Nx_M2T1NXAER-2021-2026.csv"    # aerosols


# ---------------------------------------------------------------------------
# Observation site
# ---------------------------------------------------------------------------
try:
    site_lsst = EarthLocation.of_site("Cerro Pachon")
except Exception:
    # Fallback if the astropy site registry cannot be downloaded (no network, e.g. on a cluster node)
    site_lsst = EarthLocation.from_geodetic(lon=-70.7494 * u.deg, lat=-30.2446 * u.deg, height=2663. * u.m)
