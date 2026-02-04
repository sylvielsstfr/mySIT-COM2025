import numpy as np
import pandas as pd
from typing import Dict, Any
import json

class ParameterCutSelection:
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


    #--------------------------------------------------
    # Write cuts to a json file
    #----------------------------------------------

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
