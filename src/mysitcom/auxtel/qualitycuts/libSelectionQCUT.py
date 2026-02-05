import numpy as np
import pandas as pd
from typing import Dict, Any
import json


param_ranges = {
    "alpha_0_1": (0, 5),
    "alpha_1_1": (-1, 1),
    "alpha_0_2": (0, 5),
    "gamma_0_1": (-2, 10),
    "gamma_1_1": (-5, 5),
    "gamma_2_1": (-2, 5),
    "angle [deg]": (0.1, 0.4),
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
    "ROTANGLE": (0.1, 0.3),
    "P [hPa]": (0, 2000),
}

filter_order = ["empty", "BG40_65mm_1", "OG550_65mm_1"]


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
        mask_all = np.ones(len(self.df), dtype=bool)

        for param, per_filter in cuts.items():
            if param not in self.df.columns:
                continue

            param_mask = np.ones(len(self.df), dtype=bool)

            for filt, bounds in per_filter.items():
                m = self.df[self.filter_col] == filt
                x = self.df.loc[m, param]

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

        for (target, filt), g in grouped:
            n_total = len(g)
            mask_total = np.ones(n_total, dtype=bool)

            for param, per_filter in cuts.items():
                if param not in g.columns:
                    continue
                if filt not in per_filter:
                    continue

                bounds = per_filter[filt]
                x = g[param]

                if bounds.get("min") is not None:
                    mask_total &= x >= bounds["min"]
                if bounds.get("max") is not None:
                    mask_total &= x <= bounds["max"]

            n_pass = mask_total.sum()

            rows.append({
                self.target_col: target,
                self.filter_col: filt,
                "n_total": n_total,
                "n_pass_all": n_pass,
                "frac_pass_all": n_pass / n_total if n_total > 0 else np.nan
            })

        return pd.DataFrame(rows)


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
                for filt in filter_order
            }
            for param, (vmin, vmax) in param_ranges.items()
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
