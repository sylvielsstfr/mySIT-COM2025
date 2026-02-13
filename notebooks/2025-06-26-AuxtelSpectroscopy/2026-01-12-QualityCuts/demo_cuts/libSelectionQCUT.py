import json
from typing import Any, Dict

import numpy as np
import pandas as pd

ListOfParams = [
    "alpha_0_1",
    "alpha_1_1",
    "alpha_0_2",
#    "alpha_1_2",
#    "alpha_2_2",
    "gamma_0_1",
    "gamma_1_1",
    "gamma_2_1",
    "angle [deg]",
    "alpha_pix [pix]",
    "reso [nm]",
    "shift_x [pix]",
    "shift_y [pix]",
    'MEANFWHM',
    'PIXSHIFT',
    'PSF_REG',
    'TRACE_R',
    'CHI2_FIT',
    'CHI2_FIT_norm',
    'chi2_ram_norm',
    'chi2_rum_norm',
    'D2CCD',
    'D_CCD [mm]_ram',
    'D_CCD [mm]_rum',
    'alpha_pix [pix]',
    "WINDSPD",
    "WINDDIR",
    "WINDSPDPARR",
    "WINDSPDPERP",
    "CAM_ROT",
    "ROTANGLE",
    "PARANGLE",
    "DOMEAZ",
    "AZ",
    "EL",
    "PARANGLE",
    "AIRMASS",
    "OUTTEMP",
    "OUTPRESS",
    "P [hPa]",
]

ListOfParamRanges = {
    "alpha_0_1": (0, 5),
    "alpha_1_1": (-1, 1),
    "alpha_0_2": (0, 5),
    "gamma_0_1": (-2, 10),
    "gamma_1_1": (-5, 5),
    "gamma_2_1": (-2, 5),
    "angle [deg]": (0.1, 0.4),
    "alpha_pix [pix]": (-10., 10.),
    "reso [nm]": (0, 5),
    "MEANFWHM": (0, 30),
    "PIXSHIFT": (-1, 1),
    "PSF_REG": (0, 10),
    "TRACE_R": (0, 80),
    "CHI2_FIT_norm": (0, 3),
    "chi2_ram_norm": (0, 3),
    "chi2_rum_norm": (0, 3),
    "D2CCD": (186, 189),
    "D_CCD [mm]_ram": (186, 189),
    "D_CCD [mm]_rum": (186, 189),
    "P [hPa]": (0, 3000),
}

ListOfFilterOrder = ["empty", "BG40_65mm_1", "OG550_65mm_1"]


class ParameterCutSelection:
    """Apply parameter cuts to dataframe and compute selection statistics.
    This class handles the application of quality cuts based on parameter bounds
    for different filters and computes statistics on how many observations pass
    the specified cuts.
    Attributes
    ----------
    df : pd.DataFrame
        The input dataframe containing observations and parameters.
    params : list[str]
        List of parameter column names to apply cuts on.
    id_col : str
        Name of the column containing observation identifiers.
    filter_col : str
        Name of the column containing filter information.
    target_col : str
        Name of the column containing target information.

    Return the pandas dataframe id, flag
    """
    def __init__(
        self,
        df: pd.DataFrame,
        params: list[str],
        id_col: str = "id",
        filter_col: str = "FILTER",
        target_col: str = "TARGET",
    ):
        self.df = df.copy()
        self.params = params
        self.id_col = id_col
        self.filter_col = filter_col
        self.target_col = target_col

        if id_col not in df.columns:
            self.df[id_col] = np.arange(len(df))


    # -----------------------------------------------------
    # Core: apply cuts and return row-wise flags
    # -----------------------------------------------------
    def apply_cuts(self, cuts: Dict[str, Any]) -> pd.DataFrame:
        """
        Returns a dataframe with (id, pass_all_cuts)
        """
        # container for flags for each row
        mask_all = np.ones(len(self.df), dtype=bool)

        # loop on parameters and its dictionnary of cuts bounds
        for param, per_filter in cuts.items():
            if param not in self.df.columns:
                continue

            param_mask = np.ones(len(self.df), dtype=bool)

            # loop on keys (filter) dictionnary of cuts bounds
            for filt, bounds in per_filter.items():

                # mask to select rows for that filter
                m = self.df[self.filter_col] == filt

                #extract the params values for that filter
                x = self.df.loc[m, param]

                # update the selection flag for the selected rows for that parameter
                if bounds.get("min") is not None:
                    param_mask[m] &= x >= bounds["min"]

                if bounds.get("max") is not None:
                    param_mask[m] &= x <= bounds["max"]

            mask_all &= param_mask

        return pd.DataFrame({
            self.id_col: self.df[self.id_col].values,
            "pass_all_cuts": mask_all
        })

    # -----------------------------------------------------
    # Statistics per (TARGET, FILTER)
    # -----------------------------------------------------
    def selection_statistics(self, cuts: Dict[str, Any]) -> pd.DataFrame:
        """Compute selection statistics per (TARGET, FILTER) pair.
        Applies the specified cuts to each group and calculates how many
        observations pass all cuts.
        Parameters
        ----------
        cuts : Dict[str, Any]
            Dictionary mapping parameter names to filter-specific bounds.
            Format: {param: {filter: {"min": value, "max": value}}}
        Returns
        -------
        pd.DataFrame
            Dataframe with columns: TARGET, FILTER, n_total, n_pass_all, frac_pass_all
        """
        rows = []

        grouped = self.df.groupby([self.target_col, self.filter_col])

        # Loop over each (TARGET, FILTER) group (chunck of data with same target and filter)
        for (target, filt), g in grouped:
            # Total number of observations in this group
            n_total = len(g)
            mask_total = np.ones(n_total, dtype=bool)

            # Loop over each parameter and apply the corresponding cuts for this filter
            for param, per_filter in cuts.items():
                if param not in g.columns:
                    continue
                if filt not in per_filter:
                    continue

                bounds = per_filter[filt]
                x = g[param]

                #
                if bounds.get("min") is not None:
                    mask_total &= x >= bounds["min"]
                if bounds.get("max") is not None:
                    mask_total &= x <= bounds["max"]

            # Number of observations that pass all cuts for this (TARGET, FILTER) group
            n_pass = mask_total.sum()

            #
            rows.append({
                self.target_col: target,
                self.filter_col: filt,
                "n_total": n_total,
                "n_pass_all": n_pass,
                "frac_pass_all": n_pass / n_total if n_total > 0 else np.nan
            })

        return pd.DataFrame(rows)

#-------------------------------------------------------


    def selection_statistics_inoutliers_by_param(self, cuts: Dict[str, Any]) -> pd.DataFrame:
        """
        Compute outlier statistics per (TARGET, FILTER, PARAMETER).

        Parameters
        ----------
            cuts : Dict[str, Any]
            Format: {param: {filter: {"min": value, "max": value}}}

        Returns
        -------
            pd.DataFrame
                Columns:
            TARGET, FILTER, param, n_total,
            n_below_min, n_above_max, frac_out
        """
        rows = []

        grouped = self.df.groupby([self.target_col, self.filter_col])

        # loop on chunks of targets and filters
        for (target, filt), g in grouped:

            #
            n_total = len(g)

            #
            #mask_total = np.ones(n_total, dtype=bool)

            # loop per parameter and filter
            for param, per_filter in cuts.items():
                if param not in g.columns:
                    continue
                if filt not in per_filter:
                    continue

                bounds = per_filter[filt]
                x = g[param]

                mask_pass = np.ones(n_total, dtype=bool)

                if bounds.get("min") is not None:
                    mask_pass &= x >= bounds["min"]
                if bounds.get("max") is not None:
                    mask_pass &= x <= bounds["max"]

                n_pass = mask_pass.sum()
                n_out = (~mask_pass).sum()

                n_below = 0
                n_above = 0

                if bounds.get("min") is not None:
                    n_below = (x < bounds["min"]).sum()

                if bounds.get("max") is not None:
                    n_above = (x > bounds["max"]).sum()


                #rows.append({
                #    self.target_col: target,
                #    self.filter_col: filt,
                #    "param": param,
                #    "n_total": n_total,
                #    "n_pass": n_pass,
                #    "n_out": n_out,
                #    "n_below_min": n_below,
                #    "n_above_max": n_above,
                #    "frac_pass_all": n_pass / n_total if n_total > 0 else np.nan,
                #    "frac_out": n_out / n_total if n_total > 0 else np.nan
                #})
                rows.append({
                    self.target_col: target,
                    self.filter_col: filt,
                    "param": param,
                    "n_total": n_total,
                    "n_pass": n_pass,
                    "n_out": n_out,
                    "n_below_min": (x < bounds["min"]).sum() if bounds.get("min") is not None else 0,
                    "n_above_max": (x > bounds["max"]).sum() if bounds.get("max") is not None else 0,
                    "frac_pass_all": n_pass / n_total if n_total > 0 else np.nan,
                    "frac_out": n_out / n_total if n_total > 0 else np.nan
                })

        return pd.DataFrame(rows)



#------------------------------------------------------
# Classes pour gérer les cuts : écriture, lecture, application, statistiques, plots
#-----------------------------------------------------

class ParameterCutTools:
    """
    Tools for handling parameter cuts in selection quality control.
    Provides utilities for writing and loading cuts to/from JSON files.
    """
    #--------------------------------------------------
    # Write cuts to a json file
    #----------------------------------------------

    @staticmethod
    def generate_default_paramcuts():
        """Generate a default cuts dictionary based on predefined parameter ranges."""
        cuts = {
            param: {
                filt: {"min": vmin, "max": vmax}
                for filt in ListOfFilterOrder
            }
            for param, (vmin, vmax) in ListOfParamRanges.items()
        }
        return cuts



    @staticmethod
    def write_cuts_json(
        cuts: dict,
        path: str,
        indent: int = 4,
        sort_keys: bool = True,
    ) -> None:
        """
        Write a cuts dictionary to a JSON file in a readable format.
        """

        with open(path, "w") as f:
            json.dump(
                cuts,
                f,
                indent=indent,
                sort_keys=sort_keys
            )


    # -----------------------------------------------------
    # Load cuts from a json file
    # -----------------------------------------------------
    @staticmethod
    def load_cuts_json(path: str) -> dict:
        """Load cuts from a json file

        :param path: path of the json file for cuts
        :type path: str
        :return: dictionary with cuts
        :rtype: dict
        """
        with open(path) as f:
            return json.load(f)

    #------------------------------------------------------
    # Dump json cut file into pandas dataframe
    #---------
    @staticmethod
    def cuts_to_dataframe(cuts: dict) -> pd.DataFrame:
        """Convert cuts dictionary to a pandas DataFrame for easier visualization.

        :param cuts: dictionary with cuts
        :type cuts: dict
        :return: DataFrame with columns: param, filter, min, max
        :rtype: pd.DataFrame
        """
        rows = []
        for param, per_filter in cuts.items():
            for filt, bounds in per_filter.items():
                rows.append({
                    "param": param,
                    "filter": filt,
                    "min": bounds.get("min"),
                    "max": bounds.get("max")
                })
        return pd.DataFrame(rows)
