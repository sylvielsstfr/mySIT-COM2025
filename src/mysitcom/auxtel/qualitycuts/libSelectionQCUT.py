import numpy as np
import pandas as pd
from typing import Dict, Any
import json
import matplotlib.pyplot as plt


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
    #"shift_x[pix]",
    #"shift_y[pix]",
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
#------------------------------------------------------
class ParameterCutPlotting:
    """_summary_

    :return: _description_
    :rtype: _type_
    """



    @staticmethod
    def plot_selection_fraction_by_filter(
        df_stat,
        target_color_map,
        filter_order=None,
        figsize_per_filter=(6, 0.35),
        ):
        """
        Horizontal bar plot of selection fraction per TARGET, grouped by FILTER.
        """

        if filter_order is None:
            filter_order = df_stat["FILTER"].unique()

        n_filters = len(filter_order)

        # figure height adapts to number of targets
        n_targets = df_stat["TARGET"].nunique()
        fig_height = max(4, figsize_per_filter[1] * n_targets)

        fig, axes = plt.subplots(
            1,
            n_filters,
            figsize=(figsize_per_filter[0] * n_filters, fig_height),
            sharey=True,
          )
        if n_filters == 1:
            axes = [axes]

        target_order = (
            df_stat
            .groupby("TARGET")["n_total"]
            .sum()
            .sort_values(ascending=False)
            .index
        )

        for ax, filt in zip(axes, filter_order):
            df_f = df_stat[df_stat["FILTER"] == filt].copy()

            # impose the same TARGET order for all filters
            df_f = (
                df_f
                .set_index("TARGET")
                .reindex(target_order)
                .reset_index()
            )

            # sort targets for readability
            #df_f = df_f.sort_values("frac_pass_all")

            y = np.arange(len(df_f))

            colors = [
                target_color_map.get(t, "gray")
                for t in df_f["TARGET"]
            ]

            ax.barh(
                y,
                df_f["frac_pass_all"],
                color=colors,
                edgecolor="black",
                alpha=0.9,
            )

            ax.set_title(filt)
            ax.set_xlim(0, 1.0)
            ax.grid(axis="x", alpha=0.3)

            ax.set_yticks(y)
            ax.set_yticklabels(df_f["TARGET"])

            ax.set_xlabel("Selected fraction")
            ax.invert_yaxis()

        axes[0].set_ylabel("TARGET")

        plt.tight_layout()
        return fig, axes
    

    @staticmethod
    def plot_target_param_cuts_one_filter(
        df,
        target,
        cuts,
        filter_value=None,
        target_color=None,
        ):
        """
        Barplot montrant la fraction de sélection pour chaque paramètre
        pour une TARGET donnée.
        df : dataframe original
        target : str, nom du target
        cuts : dict, structure cuts[param][filter] = {'min':..,'max':..}
        filter_value : str ou None, si on veut filtrer un filtre spécifique
        target_color : couleur pour la barre
        """

        df_t = df[df["TARGET"] == target].copy()
        if filter_value is not None:
            df_t = df_t[df_t["FILTER"] == filter_value]
        params = [p for p in cuts.keys() if p in df_t.columns]
        results = []
        for p in params:
            # applique la coupure pour ce paramètre
          if filter_value is not None:
              minv = cuts[p][filter_value].get("min")
              maxv = cuts[p][filter_value].get("max")
          else:
              # appliquer un "or" sur tous les filtres ?
              # ici on prend le min/max de la première occurrence
              minv = list(cuts[p].values())[0].get("min")
              maxv = list(cuts[p].values())[0].get("max")
          mask = pd.Series(True, index=df_t.index)
          if minv is not None:
                  mask &= df_t[p] >= minv
          if maxv is not None:
                 mask &= df_t[p] <= maxv
          n_pass = mask.sum()
          n_total = len(df_t)
          frac_pass = n_pass / n_total if n_total > 0 else 0
          results.append((p, n_pass, n_total, frac_pass))

        df_res = pd.DataFrame(results, columns=["param","n_pass","n_total","frac_pass"])

        # plot
        fig, ax = plt.subplots(figsize=(max(6, len(params)*0.6),4))
        ax.bar(df_res["param"], df_res["frac_pass"], color=target_color or "steelblue", edgecolor="black")
        ax.set_ylim(0,1)
        ax.set_ylabel("Fraction sélectionnée")
        ax.set_xlabel("Paramètre")
        ax.set_title(f"Target: {target}" + (f" | Filter: {filter_value}" if filter_value else ""))
        ax.set_xticklabels(df_res["param"], rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        return fig, df_res
    
#-----------------------------------------------------------
    @staticmethod
    def plot_target_param_cuts_multi_filters(
      df,
      target,
      cuts,
      filter_order=None,
      target_color="steelblue",
      figsize_per_subplot=(16,3)
      ):
      """
      Plot vertical barplots of fraction of selection per parameter
      for a given TARGET, one subplot per filter (aligned x-axis).
    
      df : dataframe original
      target : str
      cuts : dict, cuts[param][filter] = {'min':..,'max':..}
      filter_order : list of filters to show
      target_color : color of bars
      figsize_per_subplot : tuple(width,height) for each subplot
      """
    
      # dataframe du target
      df_t = df[df["TARGET"] == target].copy()
    
      # si filter_order non fourni, on prend tous les filtres présents
      if filter_order is None:
          filter_order = df_t["FILTER"].unique()

      n_filters = len(filter_order)
      params = [p for p in cuts.keys() if p in df_t.columns]
    
      fig, axes = plt.subplots(
          n_filters,
          1,
          figsize=(figsize_per_subplot[0], figsize_per_subplot[1]*n_filters),
          sharex=True
      )
    
      if n_filters == 1:
          axes = [axes]
    
      for ax, filt in zip(axes, filter_order):
          df_f = df_t[df_t["FILTER"] == filt].copy()
        
          results = []
          for p in params:
              if filt in cuts[p]:
                  minv = cuts[p][filt].get("min")
                  maxv = cuts[p][filt].get("max")
              else:
                  minv = None
                  maxv = None
            
              mask = pd.Series(True, index=df_f.index)
              if minv is not None:
                  mask &= df_f[p] >= minv
              if maxv is not None:
                  mask &= df_f[p] <= maxv
            
              n_pass = mask.sum()
              n_total = len(df_f)
              frac_pass = n_pass / n_total if n_total > 0 else 0
              results.append(frac_pass)
        
          ax.bar(params, results, color=target_color, edgecolor="black")
          ax.set_ylim(0,1)
          ax.set_ylabel(f"{filt}")
          ax.grid(axis="y", alpha=0.3)
    
      axes[-1].set_xticklabels(params, rotation=45, ha="right")
      axes[-1].set_xlabel("Parameter")
    
      fig.suptitle(f"Target: {target} : fraction of selected events", fontsize=14)
      plt.tight_layout(rect=[0,0,1,0.96])
    
      return fig





#

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
