"""
MagResolution_from_PWV_vs_PWV_script.py

Convert the notebook Plot2D_YMagRes_vs_xyzPWVReszPWV_LSSTfilters.ipynb
into a script that computes magnitude resolution (Y band) as a function
of PWV and PWV resolution and writes HDF5 output files.

This script does NOT produce figures by default. It loops over airmass
values (configurable) and saves an HDF5 file per airmass containing:
 - zPWV: array of airmass * PWV values
 - zdPWV: array of airmass * dPWV values
 - tresY: magnitude resolution in Y band (ny, nx)
 - tresZY: magnitude resolution for Z-Y color (ny, nx)

Usage examples:
 python MagResolution_from_PWV_vs_PWV_script.py --am-list 1.0,1.2,1.5
 python MagResolution_from_PWV_vs_PWV_script.py --am-range 1.0:2.0:0.5

Dependencies: numpy, pandas, h5py, rubinsimphot, getObsAtmo, libPhotometricCorrections

"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import h5py
import warnings

# Ignore some warnings to keep output tidy
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Attempt to import external packages; provide helpful messages if missing
try:
    from getObsAtmo import ObsAtmo
except Exception as e:
    ObsAtmo = None
    print("Warning: getObsAtmo not available. Atmospheric emulator imports may fail.")

try:
    from rubinsimphot.phot_utils import Bandpass, Sed
    from rubinsimphot.data import get_data_dir
except Exception as e:
    print("Warning: rubinsimphot not found. Sed/Bandpass imports may fail if rubinsimphot not installed.")

# Import local photometric corrections module used in the notebook
try:
    # ensure local ../lib is on sys.path so libPhotometricCorrections can be found
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
    # Import the main class expected from the local module. Avoid star-imports for clarity.
    from libPhotometricCorrections import PhotometricCorrections
except Exception:
    PhotometricCorrections = None
    print("Warning: libPhotometricCorrections not found in path. Ensure ../lib is available or adjust PYTHONPATH.")

# Provide a fallback for `filter_tagnames` that was previously available
# when using `from libPhotometricCorrections import *` in the notebook.
# Try to read a module-level `filter_tagnames` if the local module is
# importable; otherwise fall back to the standard LSST filter list.
try:
    import importlib
    libpc = importlib.import_module('libPhotometricCorrections')
    # Import common filter metadata from the local module when available
    filter_tagnames = getattr(libpc, 'filter_tagnames', ['u', 'g', 'r', 'i', 'z', 'y'])
    filter_color = getattr(libpc, 'filter_color', ['b', 'g', 'r', 'orange', 'grey', 'k'])
    FILTERWL = getattr(libpc, 'FILTERWL', None)
    filter_filenames = getattr(libpc, 'filter_filenames', None)
except Exception:
    # If the module isn't importable, default to the common LSST filters.
    filter_tagnames = ['u', 'g', 'r', 'i', 'z', 'y']
    filter_color = ['b', 'g', 'r', 'orange', 'grey', 'k']
    FILTERWL = None
    filter_filenames = None

# Utility functions (copied/adapted from the notebook)

def GenerateMultiValues(mean, sigma, size, lognorm_flag=True):
    """Generate samples for PWV with either log-normal or normal distribution.

    Parameters
    ----------
    mean : float
        Target mean value.
    sigma : float
        Standard deviation (in linear units) for the distribution.
    size : int
        Number of samples to generate.
    lognorm_flag : bool
        If True, draw from a lognormal distribution matching mean/sigma.

    Returns
    -------
    ndarray
        Array of generated samples.
    """
    if lognorm_flag:
        mu = np.log(mean**2/np.sqrt(mean**2 + sigma**2))
        sig = np.sqrt(np.log(1 + sigma**2 / mean**2))
        all_values = np.random.lognormal(mean=mu, sigma=sig, size=size)
    else:
        mu = mean
        sig = sigma
        all_values = np.random.normal(mu, sig, size=size)
    return all_values


def set_photometric_parameters(exptime, nexp, readnoise=None):
    """Create a PhotometricParameters-like object used to compute ADU.

    This mirrors the helper from the notebook. The returned object type
    should be compatible with the API expected by `Sed.calc_adu` from
    rubinsimphot (i.e. an object with the required attributes).

    Parameters
    ----------
    exptime : float
        Exposure time in seconds.
    nexp : int
        Number of exposures.
    readnoise : float or None
        Read noise in electrons/pixel (if None, default used by PhotometricParameters).

    Returns
    -------
    object
        Photometric parameters object compatible with the rest of the code.
    """
    # Try to create PhotometricParameters if available in the phot-corrections module
    try:
        # PhotometricParameters might be provided by libPhotometricCorrections
        from libPhotometricCorrections import PhotometricParameters
        photParams = PhotometricParameters(exptime=exptime, nexp=nexp, readnoise=readnoise)
        return photParams
    except Exception:
        # Fallback: create a simple namespace object with needed attributes
        class _SimpleParams:
            def __init__(self, exptime, nexp, readnoise):
                self.exptime = exptime
                self.nexp = nexp
                self.readnoise = readnoise

        return _SimpleParams(exptime, nexp, readnoise)


# Create a default photoparams object similar to the notebook default
photoparams = set_photometric_parameters(30, 1, readnoise=None)


def GetdPWVvsPWV_FromMagResolution(all_PWV_values, all_DPWV_values, magresocut=5.0,
                                    nsamples=1000, am0=1.0, oz0=300.0, tau0=0.0, beta0=1.2):
    """Find, for each PWV center value, the smallest dPWV that yields
    a magnitude resolution below `magresocut`.

    Parameters
    ----------
    all_PWV_values : array-like
        Array of PWV center values (mm).
    all_DPWV_values : array-like
        Array of dPWV candidate values (mm).
    magresocut : float
        RMS threshold in mmag to select dPWV.
    nsamples : int
        Number of PWV samples used to estimate RMS.
    am0, oz0, tau0, beta0 : floats
        Atmospheric parameters; `am0` (airmass) is now explicitly passed.

    Returns
    -------
    (np.ndarray, np.ndarray, list)
        Arrays of selected pwv centers, selected dPWV values, and list of DataFrames.
    """
    sel_dpwv = []
    sel_df = []
    sel_pwv = []

    # loop on PWV values
    for pwv0 in all_PWV_values:

        # initialize atmosphere for the typical average conditions pwv0
        pc = PhotometricCorrections(am0, pwv0, oz0, tau0, beta0)

        # create a flat SED and normalize it to have mag 20 in z
        the_sed_flat = Sed()
        the_sed_flat.set_flat_sed()
        the_sed_flat.name = 'flat'
        zmag = 20.0
        flux_norm = the_sed_flat.calc_flux_norm(zmag, pc.bandpass_total_std['z'])
        the_sed_flat.multiply_flux_norm(flux_norm)

        # loop on PWV resolution
        for dpwv in all_DPWV_values:
            if dpwv >= pwv0:
                continue

            # compute the subsamples with varying PWV
            pwv_samples = GenerateMultiValues(pwv0, dpwv, nsamples, lognorm_flag=True)
            pwv_samples = pwv_samples[np.where(np.logical_and(pwv_samples > 0.0, pwv_samples < 20.0))[0]]

            pc.CalculateMultiObs(am0, pwv_samples, oz0, tau0, beta0)

            # compute distribution for magnitude resolution
            df = CalculateMagsAndMagResolutions(pwv_samples, pc, the_sed_flat)
            df_stat = df.describe()
            rms_y = df_stat.loc["std"]["d_aduy"]
            rms_zy = df_stat.loc["std"]["d_Z-Y"]

            if rms_y <= magresocut:
                print(f"pwv0 = {pwv0:.3f} mm , dpwv = {dpwv:.3f} mm , rms_y = {rms_y:.2f} mmag rms_z-y = {rms_zy:.2f} mmag")
                sel_dpwv.append(dpwv)
                sel_df.append(df)
                sel_pwv.append(pwv0)
                break

    return np.array(sel_pwv), np.array(sel_dpwv), sel_df


def CalculateMagsAndMagResolutions(pwv_values, pc, the_sed):
    """Compute ADU-based magnitude differences and color differences for a set of PWV samples.

    This function mirrors the logic used in the notebook: it computes magnitudes or ADU
    for the (non-standard) passbands saved in the PhotometricCorrections object `pc`,
    compares to the standard ADU, and returns a DataFrame with the differences
    and derived color differences.

    Parameters
    ----------
    pwv_values : array-like
        Array of PWV sample values used to build non-standard passbands (pc should have
        computed these via `CalculateMultiObs`).
    pc : PhotometricCorrections
        Instance providing `bandpass_total_std` and `coll_bandpass_total_nonstd`.
    the_sed : Sed
        A sed object from rubinsimphot used to compute magnitudes/ADU.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: pwv, d_aduu,..., d_aduy, d_R-I, d_I-Z, d_Z-Y
        where d_aduf are differences (in mmag) relative to the standard ADU.
    """
    mag_std = {}
    adu_std = {}
    atm_bands = pc.bandpass_total_std
    # Prefer filters defined on the PhotometricCorrections instance if available,
    # otherwise fall back to the module/global `filter_tagnames` we set earlier.
    filter_tagnames_used = getattr(pc, 'filter_tagnames', filter_tagnames)

    for index, f in enumerate(filter_tagnames_used):
        mag_std[f] = the_sed.calc_mag(atm_bands[f])
        adu_std[f] = -2.5 * np.log10(the_sed.calc_adu(atm_bands[f], photoparams))

    df = pd.DataFrame(columns=["pwv", "magu", "magg", "magr", "magi", "magz", "magy", "aduu", "adug", "adur", "adui", "aduz", "aduy"])

    for idx_pwv, pwv in enumerate(pwv_values):
        mag_nonstd = {}
        adu_nonstd = {}
        atm_bands = pc.coll_bandpass_total_nonstd[idx_pwv]
        for index, f in enumerate(filter_tagnames_used):
            mag_nonstd[f] = the_sed.calc_mag(atm_bands[f])
            adu_nonstd[f] = -2.5 * np.log10(the_sed.calc_adu(atm_bands[f], photoparams))
        df.loc[idx_pwv] = [pwv, mag_nonstd["u"], mag_nonstd["g"], mag_nonstd["r"], mag_nonstd["i"], mag_nonstd["z"], mag_nonstd["y"],
                           adu_nonstd["u"], adu_nonstd["g"], adu_nonstd["r"], adu_nonstd["i"], adu_nonstd["z"], adu_nonstd["y"]]

    df = df[["pwv", "aduu", "adug", "adur", "adui", "aduz", "aduy"]]

    for index, f in enumerate(filter_tagnames):
        label_in = f'adu{f}'
        label_out = f'd_adu{f}'
        df[label_out] = (df[label_in] - adu_std[f]) * 1000.0

    df = df.drop(labels=["aduu", "adug", "adur", "adui", "aduz", "aduy"], axis=1)

    # compute relative color differences
    df["d_R-I"] = df["d_adur"] - df["d_adui"]
    df["d_I-Z"] = df["d_adui"] - df["d_aduz"]
    df["d_Z-Y"] = df["d_aduz"] - df["d_aduy"]

    return df


def MagResolutionFromPWVvsPWV_FromMagResolution(all_PWV_values, all_DPWV_values, nsamples=1000, am0=1.0,
                                                oz0=300.0, tau0=0.0, beta0=1.2):
    """Compute magnitude resolution arrays for ranges of PWV and dPWV.

    Parameters
    ----------
    all_PWV_values : array-like
        Array of PWV center values (mm).
    all_DPWV_values : array-like
        Array of PWV resolution (dPWV) values (mm).
    nsamples : int
        Number of random samples to generate per (pwv, dpwv) to estimate RMS.
    am0, oz0, tau0, beta0 : floats
        Atmospheric parameters: airmass, ozone, VAOD, Angstrom exponent.

    Returns
    -------
    zPWV : ndarray
        airmass * all_PWV_values (shape nx,)
    zdPWV : ndarray
        airmass * all_DPWV_values (shape ny,)
    tresY : ndarray
        RMS of d_aduy (shape ny, nx) in mmag
    tresZY : ndarray
        RMS of d_Z-Y (shape ny, nx) in mmag
    """
    nx = len(all_PWV_values)
    ny = len(all_DPWV_values)
    tresY = np.zeros((ny, nx))
    tresZY = np.zeros((ny, nx))

    for idx, pwv0 in enumerate(all_PWV_values):
        # initialize atmosphere for the typical average conditions pwv0
        pc = PhotometricCorrections(am0, pwv0, oz0, tau0, beta0)

        # create a flat SED and normalize it to have mag 20 in z
        the_sed_flat = Sed()
        the_sed_flat.set_flat_sed()
        the_sed_flat.name = 'flat'
        zmag = 20.0
        flux_norm = the_sed_flat.calc_flux_norm(zmag, pc.bandpass_total_std['z'])
        the_sed_flat.multiply_flux_norm(flux_norm)

        for idy, dpwv in enumerate(all_DPWV_values):
            # compute the subsamples with varying PWV
            pwv_samples = GenerateMultiValues(pwv0, dpwv, nsamples, lognorm_flag=True)
            pwv_samples = pwv_samples[np.where(np.logical_and(pwv_samples > 0.0, pwv_samples < 20.0))[0]]

            pc.CalculateMultiObs(am0, pwv_samples, oz0, tau0, beta0)

            # compute distribution for magnitude resolution
            df = CalculateMagsAndMagResolutions(pwv_samples, pc, the_sed_flat)
            df_stat = df.describe()
            rms_y = df_stat.loc["std"]["d_aduy"]
            rms_zy = df_stat.loc["std"]["d_Z-Y"]

            if idy % 10 == 0 and idx % 10 == 0:
                # print sparse progress info
                print(f"pwv = {pwv0:.1f} mm , dpwv = {dpwv:.3f} mm , res_y = {rms_y:.2f} mmag, res_zy = {rms_zy:.2f} mmag")

            tresY[idy, idx] = rms_y
            tresZY[idy, idx] = rms_zy

    zPWV = am0 * np.array(all_PWV_values)
    zdPWV = am0 * np.array(all_DPWV_values)

    return zPWV, zdPWV, tresY, tresZY


# Helper to write HDF5 results
def save_results_h5(filename_out, zPWV, zdPWV, tresY, tresZY):
    """Save arrays to an HDF5 file.

    Parameters
    ----------
    filename_out : str
        Output filename.
    zPWV, zdPWV, tresY, tresZY : ndarray
        Arrays to store.
    """
    with h5py.File(filename_out, "w") as f:
        f["zPWV"] = zPWV
        f["zdPWV"] = zdPWV
        f["tresY"] = tresY
        f["tresZY"] = tresZY


def parse_airmass_list(arg_list_str):
    """Parse a comma separated list of airmasses into floats."""
    return [float(x) for x in arg_list_str.split(",") if x.strip()]


def parse_range_spec(spec):
    """Parse a start:stop:step range specification into a list of floats.

    Example: '1.0:2.0:0.5' -> [1.0, 1.5, 2.0]
    """
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError("Range specification must be start:stop:step")
    start, stop, step = map(float, parts)
    vals = list(np.arange(start, stop + 1e-12, step))
    return vals


def main():
    parser = argparse.ArgumentParser(description="Compute magnitude resolution vs PWV and dPWV for a set of airmass values.")
    parser.add_argument("--am-list", type=str, default=None, help="Comma-separated list of airmasses, e.g. '1.0,1.2,1.5,2.0'")
    parser.add_argument("--am-range", type=str, default=None, help="Range spec start:stop:step, e.g. '1.0:2.0:0.5'")
    parser.add_argument("--nsamples", type=int, default=1000, help="Number of samples per (pwv, dpwv) to estimate RMS (default 1000)")
    parser.add_argument("--out-dir", type=str, default='results', help="Directory to save output files")
    parser.add_argument("--pwv-step", type=float, default=0.5, help="Step for main PWV grid (default 0.5 mm)")
    parser.add_argument("--dpwv-step", type=float, default=0.02, help="Step for dPWV grid (default 0.02 mm)")
    parser.add_argument("--export-contours", action='store_true', help="Export contour CSV files for tresY (optional)")
    parser.add_argument("--dry-run", action='store_true', help="Perform a dry run: compute but do not write output files")
    args = parser.parse_args()

    # build airmass list
    if args.am_list is not None:
        am_list = parse_airmass_list(args.am_list)
    elif args.am_range is not None:
        am_list = parse_range_spec(args.am_range)
    else:
        # default list
        am_list = [1.0, 1.2, 1.5, 2.0]

    os.makedirs(args.out_dir, exist_ok=True)

    # basic configuration variables (copied from notebook)
    oz0 = 300.0
    tau0 = 0.0
    beta0 = 1.2

    # PWV grids
    Delta_PWV = args.pwv_step
    all_PWV = np.arange(Delta_PWV, 20.0, Delta_PWV)

    Delta_dPWV = args.dpwv_step
    all_DPWV = np.arange(Delta_dPWV, 1.0, Delta_dPWV)

    # loop on airmass values requested
    for am0 in am_list:
        print(f"\nProcessing airmass = {am0}")
        am_num = int(am0 * 10.0)

        zPWV, zdPWV, tresY, tresZY = MagResolutionFromPWVvsPWV_FromMagResolution(
            all_PWV, all_DPWV, nsamples=args.nsamples, am0=am0, oz0=oz0, tau0=tau0, beta0=beta0
        )

        filename_out = os.path.join(args.out_dir, f"MagResolutionFromPWVvsPWV_am{am_num}_DPWV{int(Delta_PWV*100)}_DdPWV{int(Delta_dPWV*1000)}.h5")
        if args.dry_run:
            # In dry-run mode, do not write files. Print expected output paths and brief stats.
            print(f"Dry-run: would save HDF5 results to: {filename_out}")
            try:
                print(f"  tresY shape: {tresY.shape}, tresZY shape: {tresZY.shape}")
                print(f"  tresY min/max/mean: {np.nanmin(tresY):.3f}/{np.nanmax(tresY):.3f}/{np.nanmean(tresY):.3f} mmag")
            except Exception:
                print("  Unable to compute quick stats for result arrays.")
        else:
            save_results_h5(filename_out, zPWV, zdPWV, tresY, tresZY)
            print(f"Saved HDF5 results to: {filename_out}")

        # Optional: export contour lines for tresY to CSV files (one file per contour level)
        if args.export_contours and (not args.dry_run):
            try:
                import matplotlib
                # force a non-interactive backend for headless/scripted runs
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import csv

                x = zPWV
                y = zdPWV
                Z = tresY
                X, Y = np.meshgrid(x, y)

                levels = np.array([0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
                fig, ax = plt.subplots()
                cs = ax.contour(X, Y, Z, levels=levels, colors='k')

                contour_dir = os.path.join(args.out_dir, f"contours_am{am_num}")
                os.makedirs(contour_dir, exist_ok=True)

                for level, seglist in zip(cs.levels, cs.allsegs):
                    filename = os.path.join(contour_dir, f"contour_level_{level:.1f}.csv")
                    with open(filename, "w", newline="") as fcsv:
                        writer = csv.writer(fcsv)
                        writer.writerow(["x", "y"])  # header
                        for curve in seglist:
                            for (xx, yy) in curve:
                                writer.writerow([xx, yy])
                            writer.writerow([])  # blank line between curves

                    print(f"Saved contour CSV: {filename}")
                plt.close(fig)
            except Exception as exc:
                print("Failed to export contours:", exc)

    print("All done.")


if __name__ == '__main__':
    main()
