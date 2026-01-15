"""Plot functions for QCUT functions."""

# install with "pip install --user -e . " 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from pandas.api.types import is_datetime64_any_dtype
import pandas as pd

FILTER_COLORS = {
    "empty": "gray",
    "BG40_65mm_1": "blue",
    "OG550_65mm_1": "red",
}

DEFAULT_FILTER_COLOR = "purple"

def get_filter_color(filter_name):
    return FILTER_COLORS.get(filter_name, DEFAULT_FILTER_COLOR)

#------------------------------------------------------
# Functions called from QCUT01_ExploreHoloQuality.ipynb
#-------------------------------------------------------
def scatter_datetime(
    df,
    x,
    y,
    hue=None,
    palette=None,
    title=None,
    titrex=None,
    titrey=None,
    ax=None,
    figsize=(20, 8),

    # visibilité / esthétique
    s=80,
    alpha=0.5,
    edgecolor="black",
    linewidth=0.3,

    # gestion superposition
    jitter_y=0.0,
    jitter_x=0.0,
    dodge=False,

    # datetime
    date_format="%Y-%m-%d",
    major_locator="auto",

    # légende
    legend=True,
    legend_kwargs=None,
):
    """
    Scatterplot seaborn optimisé pour axes datetime et forte superposition.

    Parameters
    ----------
    jitter_x, jitter_y : float
        Amplitude du bruit ajouté aux coordonnées (en unités des axes).
        Très utile si y est discret (seq_num, index, etc.).
    dodge : bool
        Décalage horizontal par catégorie de hue (manuel).
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    data = df.copy()

    #---------------------
    # Custom color palette
    #----------------------
    palette = {
        f: FILTER_COLORS.get(f, DEFAULT_FILTER_COLOR)
        for f in data["FILTER"].unique()
    }

    # ----------------------------
    # JITTER (anti-superposition)
    # ----------------------------
    if jitter_y > 0:
        data[y] = data[y] + np.random.uniform(-jitter_y, jitter_y, size=len(data))

    if jitter_x > 0:
        data[x] = data[x] + np.random.uniform(
            -jitter_x, jitter_x, size=len(data)
        )

    # ----------------------------
    # DODGE par hue (si discret)
    # ----------------------------
    if dodge and hue is not None:
        categories = data[hue].unique()
        offsets = np.linspace(-0.15, 0.15, len(categories))
        offset_map = dict(zip(categories, offsets))
        data["_x_dodge"] = data[x] + data[hue].map(offset_map)
        x_plot = "_x_dodge"
    else:
        x_plot = x




    
    # ----------------------------
    # SCATTER
    # ----------------------------
    sns.scatterplot(
        data=data,
        x=x_plot,
        y=y,
        hue=hue,
        palette=palette,
        s=s,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
        ax=ax,
    )

    # ----------------------------
    # FORMAT DATETIME (robuste)
    # ----------------------------
    if is_datetime64_any_dtype(data[x]):
        # Matplotlib n'aime pas les tz-aware
        try:
            data[x] = data[x].dt.tz_convert(None)
        except TypeError:
            pass

        if major_locator == "auto":
            locator = mdates.AutoDateLocator()
        elif major_locator == "day":
            locator = mdates.DayLocator()
        elif major_locator == "month":
            locator = mdates.MonthLocator()
        else:
            locator = mdates.AutoDateLocator()

        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))


        

    # ----------------------------
    # LABELS
    # ----------------------------
    if title:
        ax.set_title(title)
    if titrex:
        ax.set_xlabel(titrex)
    if titrey:
        ax.set_ylabel(titrey)

    ax.tick_params(axis="x", rotation=45)

    # ----------------------------
    # LEGEND
    # ----------------------------
    if legend:
        if legend_kwargs is None:
            legend_kwargs = dict(
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0,
                frameon=True,
            )
        ax.legend(**legend_kwargs)
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    return ax
#-------------------

#-------------------
def strip_datetime(
    df,
    x,
    y,
    hue=None,
    palette=None,
    title=None,
    titrex=None,
    titrey=None,
    ax=None,
    figsize=(20, 8),

    # esthétique points
    size=8,
    alpha=1.0,
    edgecolor="black",
    linewidth=0.1,

    # jitter
    jitter=True,

    # datetime
    date_format="%Y-%m-%d",
    major_locator="auto",

    # légende
    legend=True,
    legend_kwargs=None,
):
    """
    Stripplot seaborn avec abscisse datetime (robuste UTC) et jitter.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    data = df.copy()

    #---------------------
    # Custom color palette
    #----------------------
    palette = {
        f: FILTER_COLORS.get(f, DEFAULT_FILTER_COLOR)
        for f in data["FILTER"].unique()
    }

    # ----------------------------
    # Gestion datetime (robuste)
    # ----------------------------
    if is_datetime64_any_dtype(data[x]):
        try:
            data[x] = data[x].dt.tz_convert(None)
        except TypeError:
            pass

    # ----------------------------
    # STRIPPLOT
    # ----------------------------
    sns.stripplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        size=size,
        jitter=jitter,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
        ax=ax,
    )

    # ----------------------------
    # FORMAT AXE TEMPS
    # ----------------------------
    if is_datetime64_any_dtype(data[x]):
        if major_locator == "auto":
            locator = mdates.AutoDateLocator()
        elif major_locator == "day":
            locator = mdates.DayLocator()
        elif major_locator == "month":
            locator = mdates.MonthLocator()
        else:
            locator = mdates.AutoDateLocator()

        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))

    # ----------------------------
    # LABELS
    # ----------------------------
    if title:
        ax.set_title(title)
    if titrex:
        ax.set_xlabel(titrex)
    if titrey:
        ax.set_ylabel(titrey)

    ax.tick_params(axis="x", rotation=45)

    # ----------------------------
    # LÉGENDE
    # ----------------------------
    if legend:
        if legend_kwargs is None:
            legend_kwargs = dict(
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                frameon=True,
            )
        ax.legend(**legend_kwargs)
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return ax

#--------------



#-------------

def bar_counts_by_night(
    df,
    night_col,
    filter_col,
    ax=None,
    figsize=(18, 6),

    # mode barres
    stacked=False,
    normalize=False,

    # labels
    title=None,
    xlabel=None,
    ylabel="Nombre d'entrées",

    # esthétique
    colormap=None,
    bar_width=0.8,
    rotation=45,

    # ordre explicite
    night_order=None,
    filter_order=None,

    # légende
    legend=True,
    legend_kwargs=None,
):
    """
    Bar plot du nombre d'entrées par nuit et par catégorie (filter).

    Parameters
    ----------
    stacked : bool
        Barres empilées ou côte-à-côte.
    normalize : bool
        Normalise par nuit (fractions).
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    data = df.copy()

    # ----------------------------
    # Ordre des catégories
    # ----------------------------
    if night_order is not None:
        data[night_col] = pd.Categorical(
            data[night_col], categories=night_order, ordered=True
        )

    if filter_order is not None:
        data[filter_col] = pd.Categorical(
            data[filter_col], categories=filter_order, ordered=True
        )

    # ----------------------------
    # Agrégation
    # ----------------------------
    counts = (
        data.groupby([night_col, filter_col])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    if normalize:
        counts = counts.div(counts.sum(axis=1), axis=0)


    colors = [get_filter_color(f) for f in counts.columns]
    
    # ----------------------------
    # PLOT
    # ----------------------------
    counts.plot(
        kind="bar",
        stacked=stacked,
        width=bar_width,
        #colormap=colormap,
        color=colors,
        ax=ax,
    )

    # ----------------------------
    # LABELS
    # ----------------------------
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel if xlabel else night_col)
    ax.set_ylabel(ylabel)

    ax.tick_params(axis="x", rotation=rotation)

    # ----------------------------
    # LÉGENDE
    # ----------------------------
    if legend:
        if legend_kwargs is None:
            legend_kwargs = dict(
                title=filter_col,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                frameon=True,
            )
        ax.legend(**legend_kwargs)
    else:
        ax.get_legend().remove()

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    return ax


#----------------

#----------------
def plot_dccd_chi2_vs_time(
    df,
    time_col,
    filter_col,
    dccd_col,
    chi2_col,

    # seuils / bornes
    dccd_min_fig=None,
    dccd_max_fig=None,
    dccd_min_cut=None,
    dccd_max_cut=None,
    chi2_cut=None,

    # affichage
    cmap="Set1",
    marker="+",
    lw=5,
    alpha=0.5,

    # datetime
    date_format="%y-%m-%d",

    # titres
    title_dccd="DCCD vs time",
    title_chi2="CHI2_FIT vs time",
    suptitle=None,

    # axes externes
    axs=None,
    figsize=(18, 12),
):
    """
    Trace D_CCD et CHI2_FIT vs temps sur deux panneaux verticaux.

    Parameters
    ----------
    axs : tuple(matplotlib.axes.Axes), optional
        (ax1, ax2) créés à l'extérieur. Si None, la figure est créée ici.
    """

    data = df.copy()


    # ----------------------------
    # Gestion datetime (robuste)
    # ----------------------------
    if is_datetime64_any_dtype(data[time_col]):
        try:
            data[time_col] = data[time_col].dt.tz_convert(None)
        except TypeError:
            pass

    # Codage numérique des filtres (palette discrète stable)
    filter_cat = data[filter_col].astype("category")
    filter_codes = filter_cat.cat.codes
    filter_labels = filter_cat.cat.categories

    date_form = DateFormatter(date_format)

    # ----------------------------
    # Axes
    # ----------------------------
    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    else:
        fig = axs[0].figure

    ax1, ax2 = axs

    # ============================
    # Panneau 1 : DCCD
    # ============================
    #sc1 = ax1.scatter(
    #    data[time_col],
    #    data[dccd_col],
    #    c=filter_codes,
    #    cmap=cmap,
    #    marker=marker,
    #    lw=lw,
    #    alpha=alpha,
    #)

    for f in data[filter_col].unique():
        sub = data[data[filter_col] == f]
        sc1= ax1.scatter(
            sub[time_col],
            sub[dccd_col],
            color=get_filter_color(f),  # <- couleur fixe
            marker=marker,
            lw=lw,
            alpha=alpha,
            label=f  # pour la légende
        )
    ax1.legend(title=filter_col, ncol=len(data[filter_col].unique()))
    

    if dccd_min_fig is not None and dccd_max_fig is not None:
        ax1.set_ylim(dccd_min_fig, dccd_max_fig)

    if dccd_min_cut is not None:
        ax1.axhline(dccd_min_cut, ls="-.", c="k")
    if dccd_max_cut is not None:
        ax1.axhline(dccd_max_cut, ls="-.", c="k")

    #handles, _ = sc1.legend_elements(prop="colors", alpha=alpha)
    #ax1.legend(
    #    handles,
    #    filter_labels,
    #    title=filter_col,
    #    ncols=len(filter_labels),
    #)

    ax1.set_ylabel("D_CCD [mm]")
    ax1.set_title(title_dccd)
    ax1.grid(True, alpha=0.3)

    # ============================
    # Panneau 2 : CHI2
    # ============================
    #sc2 = ax2.scatter(
    #    data[time_col],
    #    data[chi2_col],
    #    c=filter_codes,
    #    cmap=cmap,
    #    marker=marker,
    #    lw=lw,
    #    alpha=alpha,
    #)

    for f in data[filter_col].unique():
        sub = data[data[filter_col] == f]
        sc2= ax2.scatter(
            sub[time_col],
            sub[chi2_col],
            color=get_filter_color(f),
            marker=marker,
            lw=lw,
            alpha=alpha,
            label=f
        )
        
    ax2.legend(title=filter_col, ncol=len(data[filter_col].unique()))

    ax2.set_yscale("log")

    if chi2_cut is not None:
        ax2.axhline(chi2_cut, ls="-.", c="k")

    #handles, _ = sc2.legend_elements(prop="colors", alpha=alpha)
    #ax2.legend(
    #    handles,
    #    filter_labels,
    #    title=filter_col,
    #    ncols=len(filter_labels),
    #)

    ax2.set_ylabel("CHI2_FIT")
    ax2.set_xlabel("time")
    ax2.set_title(title_chi2)
    ax2.grid(True, alpha=0.3)

    # ----------------------------
    # Axe temps commun
    # ----------------------------
    ax2.xaxis.set_major_formatter(date_form)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

    # ----------------------------
    # Titre global
    # ----------------------------
    if suptitle:
        fig.suptitle(suptitle)

    if axs is None:
        fig.tight_layout()

    return fig, axs

#--------------------


#---------------------

def plot_dccd_chi2_vs_time_by_filter(
    df,
    time_col,
    filter_col,
    dccd_col,
    chi2_col,

    dccd_min_fig=None,
    dccd_max_fig=None,
    dccd_min_cut=None,
    dccd_max_cut=None,
    chi2_cut=None,

    marker="+",
    lw=5,
    alpha=0.3,

    date_format="%y-%m-%d",
    suptitle=None,
):
    """
    Génère une figure multi-lignes (une par FILTER),
    chaque ligne contenant :
      - DCCD vs time
      - CHI2 vs time
    """

   
    
    filters = df[filter_col].unique()
    n = len(filters)

    fig, axs = plt.subplots(n, 2, figsize=(18, 6 * n), sharex=False)

    if n == 1:
        axs = [axs]

    date_form = DateFormatter(date_format)

    for i, f in enumerate(filters):
        subdf = df[df[filter_col] == f]
        ax1, ax2 = axs[i]

        # --- DCCD ---
        ax1.scatter(
            subdf[time_col],
            subdf[dccd_col],
            color=get_filter_color(f),
            marker=marker,
            lw=lw,
            alpha=alpha,
        )

        if dccd_min_fig is not None and dccd_max_fig is not None:
            ax1.set_ylim(dccd_min_fig, dccd_max_fig)
        if dccd_min_cut is not None:
            ax1.axhline(dccd_min_cut, ls="-.", c="k")
        if dccd_max_cut is not None:
            ax1.axhline(dccd_max_cut, ls="-.", c="k")

        ax1.set_title(f"{f} – DCCD vs time")
        ax1.set_ylabel("D_CCD [mm]")
        ax1.set_xlabel("time")
        ax1.xaxis.set_major_formatter(date_form)
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

        # --- CHI2 ---
        ax2.scatter(
            subdf[time_col],
            subdf[chi2_col],
            color=get_filter_color(f),
            marker=marker,
            lw=lw,
            alpha=alpha,
        )

        ax2.set_yscale("log")
        if chi2_cut is not None:
            ax2.axhline(chi2_cut, ls="-.", c="k")

        ax2.set_title(f"{f} – CHI2_FIT")
        ax2.set_ylabel("CHI2_FIT")
        ax2.set_xlabel("time")
        ax2.xaxis.set_major_formatter(date_form)
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()
    return fig, axs



    
