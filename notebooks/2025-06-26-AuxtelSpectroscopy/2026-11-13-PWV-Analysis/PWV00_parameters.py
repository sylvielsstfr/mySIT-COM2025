# Centralize config parameters, paths to input files, selection cuts and selection functions
# for the PWV analysis notebooks (PWV01, PWV02, ...).
#
# author : Sylvie Dagoret-Campagne
# affiliation : IJCLab/IN2P3/CNRS
# Creation date : 2025-09-20
# last update : 2025-12-14 : run_v12 (2025 data)
# last update : 2026-11-13 : adapted to the merged files produced in
#                            ../2026-11-13-SpectractorExtraction-FromButler (runs 2023-2026 from Corentin Ravoux)
#
# Main changes with respect to 2025-09-16-PWV-Analysis/PWV00_parameters.py :
#   - input files are the 3 merged files of 2026-11-13-SpectractorExtraction-FromButler
#   - butler repo / collection bookkeeping removed (not needed for the analysis)
#   - column names changed : "PWV [mm]_x" -> "PWV [mm]_ram" or "PWV [mm]_rum"
#     (selected by PWV_METHOD, see below). Use PWV_COL, PWV_ERR_COL, ... in the notebooks.
#   - filters in the data are now : empty, BG40_65mm_1, OG550_65mm_1 (no SDSSr, no FELH0600)
#   - Merra2 files extended up to 2026-10
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
# Filters present in the 2026 merged files : empty, BG40_65mm_1, OG550_65mm_1
PWV_FILTER_LIST = ["empty", "OG550_65mm_1"]
PWV_FILTEROG550_LIST = ["OG550_65mm_1"]


# ---------------------------------------------------------------------------
# Column names of the fit results in the merged files
# ---------------------------------------------------------------------------
# The merged files contain two sets of atmospheric fit parameters, with suffix "_ram" and "_rum" :
#   "rum" : parameters fitted on the 1D spectrum (projection on the dispersion axis) in Spectractor
#   "ram" : parameters fitted on the spectrogram (2D CCD image) in Spectractor
# NB : in the Jan-Jul 2026 file, "PWV [mm]_err_ram" has extreme outliers (up to 1e9 mm),
#      while "PWV [mm]_err_rum" stays below ~130 mm : a cut on the error may be needed with "ram".
PWV_METHOD = "rum"      # "ram" or "rum"

PWV_COL = f"PWV [mm]_{PWV_METHOD}"
PWV_ERR_COL = f"PWV [mm]_err_{PWV_METHOD}"
OZONE_COL = f"ozone [db]_{PWV_METHOD}"
OZONE_ERR_COL = f"ozone [db]_err_{PWV_METHOD}"
VAOD_COL = f"VAOD_{PWV_METHOD}"
VAOD_ERR_COL = f"VAOD_err_{PWV_METHOD}"
DCCD_FIT_COL = f"D_CCD [mm]_{PWV_METHOD}"

# Other columns used by the cuts
CHI2_COL = "CHI2_FIT"
D2CCD_COL = "D2CCD"
EXPTIME_COL = "EXPTIME"
TARGET_COL = "TARGET"
FILTER_COL = "FILTER"
TIME_COL = "DATE-OBS"

POLARIZED_TARGET = "HD185975"


def DumpConfig():
    """
    Dump the configuration chosen here
    """
    print("**************************************************************************")
    print("*                         DumpConfig()                                   *")
    print(f" \t - version_run = {version_run}")
    print(f" \t - annotation tag = {tag}")
    print(f" \t - merged file = {atmfilename}")
    print(f" \t - fit method suffix = _{PWV_METHOD} -> PWV column = '{PWV_COL}'")
    print(f" \t - PWV filters = {PWV_FILTER_LIST}")
    print("**************************************************************************")


# ---------------------------------------------------------------------------
# Selection parameters for cuts
# ---------------------------------------------------------------------------
# Median of D2CCD in the 2026 data is ~187.15 mm (5%-95% : 186.6-187.7 mm)
DCCDMINFIG = 185.0
DCCDMAXFIG = 190.0
DCCDMINCUT = 186.7
DCCDMAXCUT = 188.0
CHI2CUT = 50.
EXPTIMECUT = 20.0     # NB : all exposures are 30 s in the 2026 merged files
PWVMINCUT = 0.
PWVMAXCUT = 20.
OZMINCUT = 0.
OZMAXCUT = 650.0


def getSelectionCut(df_spec, chi2max=CHI2CUT, pwvmin=PWVMINCUT, pwvmax=PWVMAXCUT,
                    ozmin=OZMINCUT, ozmax=OZMAXCUT):
    """Standard quality selection (all targets). ozmin/ozmax kept for API compatibility (not applied)."""
    cut = ((df_spec[CHI2_COL] < chi2max)
           & (df_spec[PWV_COL] > pwvmin) & (df_spec[PWV_COL] < pwvmax)
           & (df_spec[D2CCD_COL] > DCCDMINCUT) & (df_spec[D2CCD_COL] < DCCDMAXCUT)
           & (df_spec[EXPTIME_COL] > EXPTIMECUT))
    return cut


def getSelectionCutNoPolar(df_spec, chi2max=CHI2CUT, pwvmin=PWVMINCUT, pwvmax=PWVMAXCUT,
                           ozmin=OZMINCUT, ozmax=OZMAXCUT):
    """Standard selection excluding the polarized star HD185975."""
    return getSelectionCut(df_spec, chi2max, pwvmin, pwvmax, ozmin, ozmax) & (df_spec[TARGET_COL] != POLARIZED_TARGET)


def getSelectionCutWithPolar(df_spec, chi2max=CHI2CUT, pwvmin=PWVMINCUT, pwvmax=PWVMAXCUT,
                             ozmin=OZMINCUT, ozmax=OZMAXCUT):
    """Standard selection restricted to the polarized star HD185975."""
    return getSelectionCut(df_spec, chi2max, pwvmin, pwvmax, ozmin, ozmax) & (df_spec[TARGET_COL] == POLARIZED_TARGET)


# ---------------------------------------------------------------------------
# Photometric repeatability
# ---------------------------------------------------------------------------
FLAG_CORRECTFOR_PWV_REPEAT = True
FLAG_CORRECTFOR_PWV_REPEAT_RATIO = False

#SIGMA_PWV_REPEAT = 0.25
SIGMA_PWV_REPEAT = 0.10      # TO BE RE-ESTIMATED with the 2026 data (see PWV02 repeatability notebooks)
FACTORERR_PWV_REPEAT = 10.


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


def get_astronomical_midnight(location: EarthLocation, date, n_grid=1000):
    """
    Lower transit of the Sun (minimum altitude) for a given date and site, with pure astropy.

    Parameters
    ----------
    location : EarthLocation
        Observation site (lat, lon, height)
    date : str, datetime, datetime.date or astropy Time
        Reference date (e.g. "2025-09-23")
    n_grid : int
        Number of points over 24 h used to locate the minimum

    Returns
    -------
    datetime
        UTC time when the Sun altitude is minimal
    """
    if isinstance(date, datetime):
        t = Time(date, scale="utc")
    elif hasattr(date, "strftime"):  # e.g. datetime.date
        t = Time(date.strftime("%Y-%m-%d"), scale="utc")
    elif not isinstance(date, Time):
        t = Time(date, scale="utc")
    else:
        t = date

    date = t + 1 * u.day

    # ~24 h interval around the date
    t0 = date - 12 * u.hour
    t1 = date + 12 * u.hour

    times = Time(np.linspace(t0.jd, t1.jd, n_grid), format='jd', scale='utc')

    sun = get_sun(times)
    aa = AltAz(obstime=times, location=location)
    sun_altaz = sun.transform_to(aa)

    idx_min = np.argmin(sun_altaz.alt)
    t_min = times[idx_min]
    return t_min.to_datetime()
