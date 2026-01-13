#!/usr/bin/env python3
"""
EXTR_viewSpectractorResults_December2025_repomain-new.py

Standalone script extracted from the Jupyter notebook:
EXTR_viewSpectractorResults_December2025_repomain-new.ipynb

Purpose:
- Query a Butler repo for 'spectractorSpectrum' datasets,
- extract Libradtran fit parameters (spectrum & spectrogram),
- assemble a DataFrame and save a Numpy record array (np.save) as in the original notebook.
- Optionally produce non-interactive plots (saved as PNG).
- Default: do not display figures.

Notes:
- If available, imports BUTLER00_parameters for repo/collection defaults; otherwise they can be provided via CLI.
- Requires conda_py313 environment and Rubin dependencies (Butler, rubinsimphot as needed).
"""

from __future__ import annotations
import argparse
import os
import sys
import json
import logging
from functools import partial
import multiprocessing as mp
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# If you have local libs in /notebooks/.../lib, add them robustly
import os as _os
_script_dir = os.path.dirname(os.path.abspath(__file__))
_lib_dir = os.path.join(_script_dir, "..", "lib")
_lib_dir = os.path.normpath(_lib_dir)
if os.path.exists(_lib_dir) and _lib_dir not in sys.path:
    sys.path.append(_lib_dir)

# try importing Butler parameters local module
try:
    from BUTLER00_parameters import *
except Exception:
    # If that module is not available, we'll fallback to CLI-provided args
    pass

# import Butler related only when we need it (to reduce import errors when CLI checks)
try:
    import lsst.daf.butler as dafButler
except Exception:
    dafButler = None


# Optional extra module used in the notebook:
try:
    import getCalspec
except Exception:
    getCalspec = None

# Matplotlib headless mode (no UI)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def configure_logging(verbose: bool = False) -> None:
    """Set logging to INFO or DEBUG depending on the verbose flag."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for script operation."""
    parser = argparse.ArgumentParser(description="Extract spectractor/Libradtran parameters from a Butler repo.")
    parser.add_argument("--repo", default=None, help="Butler repo path (defaults to repo from parameters or /repo/main).")
    parser.add_argument("--embargo", action="store_true", help="Use embargo repo (/repo/embargo).")
    parser.add_argument("--version-run", default=None, help="Version run key for butlerusercollectiondict (defaults from parameters when available).")
    parser.add_argument("--collection", default=None, help="Explicit collection name to use.")
    parser.add_argument("--out-file", default=None, help="Output filename (np.save) to hold the final record array.")
    parser.add_argument("--n-workers", type=int, default=16, help="Number of worker processes for pool.")
    parser.add_argument("--filter-instrument", default="LATISS", help="Instrument filter name for the dataset query.")
    parser.add_argument("--dry-run", action="store_true", help="If set, do not write output file.")
    parser.add_argument("--plot", action="store_true", help="If set, save summary plots to a results directory.")
    parser.add_argument("--plot-dir", default="plots", help="Directory to save plots if --plot is set.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    parser.add_argument("--only-check", action="store_true", help="Only check presence of datasets and print counts then exit.")
    args = parser.parse_args()
    return args


def initialize_butler(repo: str | None, embargo: bool):
    """Initialize a Butler from the repo path (or default)."""
    if repo is None:
        if 'FLAG_REPO_EMBARGO' in globals() and FLAG_REPO_EMBARGO:
            repo = "/repo/embargo"
        else:
            repo = "/repo/main"
    if embargo:
        repo = "/repo/embargo"

    if dafButler is None:
        raise RuntimeError("lsst.daf.butler is not importable. Activate the right environment (conda_py313).")

    logging.info("Initializing Butler with repo: %s", repo)
    butler = dafButler.Butler(repo)
    return butler


def get_collection_from_parameters(version_run_arg: str | None, collection_arg: str | None):
    """Get collection name from passed args or from the imported parameters if available."""
    if collection_arg:
        logging.info("Using explicit collection: %s", collection_arg)
        return collection_arg

    if version_run_arg is None and 'version_run' in globals():
        version_run_arg = version_run

    if version_run_arg and 'butlerusercollectiondict' in globals() and version_run_arg in butlerusercollectiondict:
        return butlerusercollectiondict[version_run_arg]

    if 'collection_validation' in globals():
        return collection_validation

    raise RuntimeError("Collection not found. Provide --collection or ensure BUTLER00_parameters defines it.")


def query_dataset_and_records(butler, my_collection, instrument='LATISS'):
    """
    Query datasets and records:
    - datasetRefs: dataset references for 'spectractorSpectrum'
    - records: dimension records for 'visit' with that dataset
    - refs: unique dataset refs
    """
    registry = butler.registry
    where = f"instrument='{instrument}'"
    datasetRefs = registry.queryDatasets(datasetType='spectractorSpectrum', collections=[my_collection], where=where)
    records = list(registry.queryDimensionRecords('visit', datasets='spectractorSpectrum', where=where,  collections=[my_collection]))
    refs = list(set(registry.queryDatasets('spectractorSpectrum',  where=where,  collections=[my_collection])))
    logging.info("Number of records: %d", len(records))
    logging.info("Number of datasetRefs: %d", len(datasetRefs))
    return datasetRefs, records, refs


def extract_results(ref, my_collection, butler):
    """
    Given a dataset ref, fetch:
    - spectractorSpectrum
    - spectrumLibradtranFitParameters
    - spectrogramLibradtranFitParameters
    Return header, spectrumParams, spectrogramParams
    """
    try:
        spec = butler.get(
            "spectractorSpectrum",
            visit=ref.dataId["visit"],
            collections=[my_collection],
            detector=0,
            instrument="LATISS",
        )
        header = spec.header.copy()
        header["ID"] = ref.dataId["visit"]

        params_spectrum = butler.get(
            "spectrumLibradtranFitParameters",
            visit=ref.dataId["visit"],
            collections=[my_collection],
            detector=0,
            instrument="LATISS",
        )
        params_spectrogram = butler.get(
            "spectrogramLibradtranFitParameters",
            visit=ref.dataId["visit"],
            collections=[my_collection],
            detector=0,
            instrument="LATISS",
        )
        return header, params_spectrum, params_spectrogram
    except Exception as e:
        # Return None to mark missing dataset/exception
        logging.debug("Exception while extracting %s: %s", ref, str(e))
        return None


def pool_extract(refs, butler, my_collection, n_workers=16):
    """
    Extract result via a multiprocess Pool or sequentially if n_workers=1.
    Returns a list of results (header, spectrum params, spectrogram params) or None entries.
    """
    extract_func = partial(extract_results, my_collection=my_collection, butler=butler)
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            results = list(pool.imap(extract_func, refs))
    else:
        results = [extract_func(r) for r in refs]
    return results


def concatenate_results_and_save(results, records, file_save, dry_run=False):
    """
    From results (list of tuples or None) and records, create the Pandas DataFrame and save rec as np.save
    """
    headers = []
    params_spectrum = []
    params_spectrogram = []

    nskip = 0
    for res in results:
        if res is None:
            nskip += 1
        else:
            headers.append(res[0])
            params_spectrum.append(res[1])
            params_spectrogram.append(res[2])
    logging.info("Skipping %d failed results", nskip)

    if not headers:
        raise RuntimeError("No valid headers extracted; aboring.")

    # Build column lists
    columns_spectrum = ["id"]
    for h in headers[0]:
        if "COMMENT" in h or "EXTNAME" in h:
            continue
        if "LBDAS_T" in h or "PSF_P_T" in h or "AMPLIS_T" in h:
            continue
        if "UNIT" in h:
            continue
        if "SIMPLE" in h:
            continue
        columns_spectrum.append(h)

    columns_spectrogram_bestfit = []
    for key in params_spectrogram[0].labels:
        columns_spectrogram_bestfit.append(key)
        columns_spectrogram_bestfit.append(key + "_err")

    columns_spectrum_bestfit = []
    for key in params_spectrum[0].labels:
        columns_spectrum_bestfit.append(key)
        columns_spectrum_bestfit.append(key + "_err")

    # Compose dataframes
    df1 = pd.DataFrame(columns=columns_spectrum)
    for k, header in enumerate(headers):
        n = int(records[k].id)
        row = {"id": n}
        for h in header:
            if h in columns_spectrum:
                row[h] = header[h]
        df1.loc[len(df1)] = row

    df2 = pd.DataFrame(columns=columns_spectrogram_bestfit)
    for k, p in enumerate(params_spectrogram):
        n = int(records[k].id)
        row = {"id": n}
        for i, key in enumerate(p.labels):
            row[key] = p.values[i]
            row[key + "_err"] = p.err[i]
        df2.loc[len(df2)] = row

    df3 = pd.DataFrame(columns=columns_spectrum_bestfit)
    for k, p in enumerate(params_spectrum):
        n = int(records[k].id)
        row = {"id": n}
        for i, key in enumerate(p.labels):
            row[key] = p.values[i]
            row[key + "_err"] = p.err[i]
        df3.loc[len(df3)] = row

    # Merge dataframes
    df = pd.merge(df1, df2, left_index=True, right_index=True)
    df = pd.merge(df, df3, left_index=True, right_index=True)

    df.set_index("DATE-OBS", inplace=True)
    df.index = pd.to_datetime(df.index, format="ISO8601")
    df.sort_index(inplace=True)

    rec = df.to_records()
    logging.info("Final record shape: %s", rec.shape)
    if not dry_run:
        logging.info("Saving results to %s", file_save)
        np.save(file_save, rec, allow_pickle=True)
    else:
        logging.info("Dry-run: skipping file save for %s", file_save)
    return rec, df


def plot_summary(rec, df, outdir="plots", prefix="auxtel_summary"):
    """
    Save some summary plots as PNG in a directory (no GUI).
    """
    os.makedirs(outdir, exist_ok=True)

    # columns to plot similar to the notebook
    columns_to_plot = ["D2CCD", "PIXSHIFT", "PSF_REG", "CHI2_FIT", "OUTPRESS", "OUTTEMP", "alpha_0_2", "TARGETX", "TARGETY"]
    for col in columns_to_plot:
        if col not in df.columns:
            continue
        if len(col.split("_")) > 1:
            col_err = "_".join(col.split("_")[:-1]) + "_err_" + col.split("_")[-1]
        else:
            col_err = col + "_err"

        plt.figure(figsize=(10, 4))
        if col_err in df.columns:
            plt.errorbar(rec["DATE-OBS"], rec[col], yerr=rec[col_err], linestyle="none", marker="+")
        else:
            plt.plot(rec["DATE-OBS"], rec[col], linestyle="none", marker="+")

        plt.ylim((0.9 * np.nanmin(rec[col]), 1.1 * np.nanmax(rec[col])))
        if "PSF_REG" in col:
            plt.yscale("log")
        plt.grid()
        plt.title(col)
        plt.gcf().autofmt_xdate()

        outpng = os.path.join(outdir, f"{prefix}_{col}.png")
        plt.savefig(outpng, dpi=150, bbox_inches="tight")
        plt.close()


def main():
    """Main entry point."""
    args = parse_args()
    configure_logging(args.verbose)

    # If local parameters define repo/collection/others, prefer them unless CLI overrides are present
    repo = args.repo
    if repo is None and 'FLAG_REPO_EMBARGO' in globals() and FLAG_REPO_EMBARGO:
        repo = "/repo/embargo"
    if repo is None:
        repo = "/repo/main"

    butler = initialize_butler(repo, embargo=args.embargo)
    my_collection = get_collection_from_parameters(version_run_arg=args.version_run, collection_arg=args.collection)

    _, records, refs = query_dataset_and_records(butler, my_collection, instrument=args.filter_instrument)

    if args.only_check:
        print(f"Number of records: {len(records)}. Exiting by --only-check.")
        return

    # Using a results filename default similar to the notebook naming
    out_file = args.out_file
    if out_file is None:
        # fallback composition if not provided
        collection_name = my_collection.replace("/", "_")
        out_file = f"auxtel_run_{collection_name}_v1.npy"

    # Extract with multiple workers
    logging.info("Starting extraction for collection %s with %d workers", my_collection, args.n_workers)
    results = pool_extract(refs, butler, my_collection, n_workers=args.n_workers)

    # Build and save the final rec / dataframe
    rec, df = concatenate_results_and_save(results, records, out_file, dry_run=args.dry_run)

    #if args.plot:
    #    plot_dir = args.plot_dir
    #    plot_summary(rec, df, outdir=plot_dir)
    #    logging.info("Plots saved to %s", plot_dir)

    logging.info("Done.")


if __name__ == "__main__":
    main()