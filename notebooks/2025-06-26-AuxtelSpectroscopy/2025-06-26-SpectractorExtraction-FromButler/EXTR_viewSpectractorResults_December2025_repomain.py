#!/usr/bin/env python3
"""
EXTR_viewSpectractorResults_December2025_repomain.py

Standalone script extracted from the notebook:
EXTR_viewSpectractorResults_December2025_repomain.ipynb

- Default: do not show figures (headless backend)
- Optional: save summary PNG plots with --plot
- Uses BUTLER00_parameters if present (fallback to CLI args)
"""

from __future__ import annotations
import argparse
import os
import sys
import logging
from datetime import datetime
from functools import partial
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm

# robust path for local lib (same directory structure as notebooks)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_lib_dir = os.path.normpath(os.path.join(_script_dir, "..", "lib"))
if os.path.exists(_lib_dir) and _lib_dir not in sys.path:
    sys.path.append(_lib_dir)

# Try to import optional config parameters from local file
try:
    from BUTLER00_parameters import *
except Exception:
    # If not present we'll use CLI defaults or required args
    pass

# Butler import lazy check
try:
    import lsst.daf.butler as dafButler
except Exception:
    dafButler = None

# Matplotlib headless mode (no UI)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional getCalspec used in notebook
try:
    import getCalspec
except Exception:
    getCalspec = None


def configure_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description="Extract spectractor/Libradtran parameters from Butler and save a record array.")
    p.add_argument("--repo", default=None, help="Butler repo path (defaults to param or /repo/main)")
    p.add_argument("--embargo", action="store_true", help="Use embargo repo (/repo/embargo)")
    p.add_argument("--version-run", default=None, help="version run key to find collection in BUTLER00_parameters")
    p.add_argument("--collection", default=None, help="explicit collection to use (string)")
    p.add_argument("--out-file", default=None, help="Output numpy filename, e.g. results.npy")
    p.add_argument("--n-workers", type=int, default=16, help="Number of worker processes (parallel)")
    p.add_argument("--plot", action="store_true", help="Save summary PNG plots to --plot-dir")
    p.add_argument("--plot-dir", default="plots", help="Directory for PNG output when --plot is given")
    p.add_argument("--dry-run", action="store_true", help="Perform run without saving npy file")
    p.add_argument("--only-check", action="store_true", help="Only count and print dataset records, then exit")
    p.add_argument("--instrument", default="LATISS", help="Instrument filter (default LATISS)")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args()


def initialize_butler(repo: str | None, embargo: bool) -> "Butler":
    """Initialize a Butler instance for a repo path."""
    repo_path = repo
    if repo_path is None:
        in_params = globals()
        if "FLAG_REPO_EMBARGO" in in_params and in_params["FLAG_REPO_EMBARGO"]:
            repo_path = "/repo/embargo"
        else:
            repo_path = "/repo/main"
    if embargo:
        repo_path = "/repo/embargo"
    if dafButler is None:
        raise ImportError("lsst.daf.butler not importable. Activate the conda_py313 environment with Butler installed.")
    logging.info("Initializing Butler with repo: %s", repo_path)
    butler = dafButler.Butler(repo_path)
    return butler


def choose_collection(version_run_arg: str | None, collection_arg: str | None):
    """Return the collection name (string). Prefer CLI override, else look in import parameters."""
    if collection_arg:
        return collection_arg
    if version_run_arg is None and "version_run" in globals():
        version_run_arg = globals()["version_run"]
    if version_run_arg and "butlerusercollectiondict" in globals() and version_run_arg in globals()["butlerusercollectiondict"]:
        return globals()["butlerusercollectiondict"][version_run_arg]
    if "collection_validation" in globals():
        return globals()["collection_validation"]
    raise RuntimeError("Collection not resolved via CLI or parameters; provide --collection or set BUTLER00_parameters.")


def query_dataset_and_records(butler, collection: str, instrument: str = "LATISS"):
    """Query records and refs for datasetType 'spectractorSpectrum'."""
    registry = butler.registry
    where = f"instrument='{instrument}'"
    datasetRefs = registry.queryDatasets(datasetType="spectractorSpectrum", collections=[collection], where=where)
    records = list(registry.queryDimensionRecords("visit", datasets="spectractorSpectrum", where=where, collections=[collection]))
    refs = list(set(registry.queryDatasets("spectractorSpectrum", where=where, collections=[collection])))
    logging.info("Found %d records, %d unique refs", len(records), len(refs))
    return datasetRefs, records, refs


def extract_single_ref(ref, collection, butler):
    """Fetch the spectractor spectrum and libradtran parameters for a single ref."""
    try:
        spec = butler.get("spectractorSpectrum", visit=ref.dataId["visit"], collections=[collection], detector=0, instrument="LATISS")
        header = spec.header.copy()
        header["ID"] = ref.dataId["visit"]

        params_spectrum = butler.get("spectrumLibradtranFitParameters", visit=ref.dataId["visit"], collections=[collection], detector=0, instrument="LATISS")
        params_spectrogram = butler.get("spectrogramLibradtranFitParameters", visit=ref.dataId["visit"], collections=[collection], detector=0, instrument="LATISS")
        return header, params_spectrum, params_spectrogram
    except Exception as e:
        logging.debug("Failed to extract ref %s: %s", ref.dataId, str(e))
        return None


def pool_extract(refs, butler, collection, n_workers):
    """Parallel extraction via multiprocessing pool (imap to maintain order)."""
    extract_func = partial(extract_single_ref, collection=collection, butler=butler)
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            results = list(pool.imap(extract_func, refs))
    else:
        results = [extract_func(r) for r in refs]
    return results


def build_dataframe_and_save(results, records, out_file, dry_run=False):
    """
    Build DataFrame from extracted results and save as numpy record.
    - results: list of (header, params_spectrum, params_spectrogram) or None entries
    - records: list of dimension records corresponding to dataset refs
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
    logging.info("Records skipped because of errors: %d", nskip)
    if len(headers) == 0:
        raise RuntimeError("No successful results to process.")

    # columns from header
    columns_spectrum = ["id"]
    for h in headers[0]:
        if "COMMENT" in h or "EXTNAME" in h:
            continue
        if "LBDAS_T" in h or "PSF_P_T" in h or "AMPLIS_T" in h:
            continue
        if "UNIT" in h or "SIMPLE" in h:
            continue
        columns_spectrum.append(h)

    # columns for spectrogram best fit
    columns_spectrogram_bestfit = []
    for key in params_spectrogram[0].labels:
        columns_spectrogram_bestfit.append(key)
        columns_spectrogram_bestfit.append(key + "_err")

    # columns for spectrum best fit
    columns_spectrum_bestfit = []
    for key in params_spectrum[0].labels:
        columns_spectrum_bestfit.append(key)
        columns_spectrum_bestfit.append(key + "_err")

    # build df1 from headers
    df1 = pd.DataFrame(columns=columns_spectrum)
    for k, header in enumerate(headers):
        n = int(records[k].id)
        row = {"id": n}
        for h in header:
            if h in columns_spectrum:
                row[h] = header[h]
        df1.loc[len(df1)] = row

    # df2 spectrogram best-fit rows
    df2 = pd.DataFrame(columns=columns_spectrogram_bestfit)
    for k, p in enumerate(params_spectrogram):
        n = int(records[k].id)
        row = {"id": n}
        for i, key in enumerate(p.labels):
            row[key] = p.values[i]
            row[key + "_err"] = p.err[i]
        df2.loc[len(df2)] = row

    # df3 spectrum best-fit rows
    df3 = pd.DataFrame(columns=columns_spectrum_bestfit)
    for k, p in enumerate(params_spectrum):
        n = int(records[k].id)
        row = {"id": n}
        for i, key in enumerate(p.labels):
            row[key] = p.values[i]
            row[key + "_err"] = p.err[i]
        df3.loc[len(df3)] = row

    df = pd.merge(df1, df2, left_index=True, right_index=True)
    df = pd.merge(df, df3, left_index=True, right_index=True)

    df.set_index("DATE-OBS", inplace=True)
    df.index = pd.to_datetime(df.index, format="ISO8601")
    df.sort_index(inplace=True)

    rec = df.to_records()
    logging.info("Resulting rec shape: %s", rec.shape)
    if not dry_run:
        logging.info("Saving %s", out_file)
        np.save(out_file, rec, allow_pickle=True)
    else:
        logging.info("Dry-run: skipping save")
    return rec, df


def run_plots_for_summary(rec, df, outdir="plots"):
    """Save a set of summary PNGs matching the notebook plots (no GUI)."""
    os.makedirs(outdir, exist_ok=True)
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
        outpng = os.path.join(outdir, f"summary_{col}.png")
        plt.savefig(outpng, dpi=150, bbox_inches="tight")
        plt.close()


def main():
    args = parse_args()
    configure_logging(args.verbose)

    repo = args.repo
    if repo is None and "FLAG_REPO_EMBARGO" in globals() and globals()["FLAG_REPO_EMBARGO"]:
        repo = "/repo/embargo"
    if repo is None:
        repo = "/repo/main"

    butler = initialize_butler(repo, embargo=args.embargo)
    my_collection = choose_collection(version_run_arg=args.version_run, collection_arg=args.collection)
    logging.info("Using collection: %s", my_collection)

    datasetRefs, records, refs = query_dataset_and_records(butler, my_collection, instrument=args.instrument)

    if args.only_check:
        logging.info("Only-check mode: found %d records", len(records))
        return

    # Filename default similar to the notebook
    if args.out_file is None:
        collection_name = my_collection.replace("/", "_")
        out_file = f"auxtel_run_{collection_name}_v1.npy"
    else:
        out_file = args.out_file

    logging.info("Extracting %d refs with %d workers...", len(refs), args.n_workers)
    results = pool_extract(refs, butler, my_collection, n_workers=args.n_workers)

    logging.info("Building DataFrame and saving results...")
    rec, df = build_dataframe_and_save(results, records, out_file, dry_run=args.dry_run)

    if args.plot:
        run_plots_for_summary(rec, df, args.plot_dir)

    logging.info("Done.")

if __name__ == "__main__":
    main()