"""Plot functions for QCUT functions."""

# install with "pip install --user -e . "

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.dates import DateFormatter
from pandas.api.types import is_datetime64_any_dtype
import pandas as pd
from pprint import pprint

FILTER_COLORS = {
    "empty": "gray",
    "BG40_65mm_1": "blue",
    "OG550_65mm_1": "red",
}

DEFAULT_FILTER_COLOR = "purple"
DEFAULT_TARGET_COLOR ="lightgrey"

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


#----------------------

#----------------------

def stripplot_target_vs_time(
    df,
    time_col="Time",
    target_col="TARGET",
    seq_col="seq_num",
    ax=None,
    tag="",
    size=10,
    alpha=1.0,
    edgecolor="black",
    linewidth=0.1,
    jitter=True,
    palette=None
):
    """
    Stripplot de TARGET vs Time avec couleurs distinctes par TARGET.
    Peut recevoir un axe externe ax.
    """

    data = df.copy()
    data["TARGET_seq"] = data[target_col].astype(str) + "_" + data[seq_col].astype(str)

    # Création dynamique de la palette si non fournie
    targets = data[target_col].unique()
    n_targets = len(targets)

    if palette is None:
        # Générer n couleurs distinctes
        cmap = plt.get_cmap("tab20")  # tab20 contient 20 couleurs
        if n_targets > 20:
            cmap = plt.get_cmap("hsv")  # HSV permet de générer n couleurs très distinctes
        palette = {t: mcolors.to_hex(cmap(i / n_targets)) for i, t in enumerate(targets)}

    # Création de la figure si ax non fourni
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 12))
    else:
        fig = ax.figure

    sns.stripplot(
        data=data,
        x=time_col,
        y=target_col,
        hue=target_col,
        palette=palette,
        size=size,
        jitter=jitter,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
        ax=ax
    )

    ax.set_title(f"Auxtel Holo observations wrt date and target {tag}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Target")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True)
    ax.legend(title="Target", bbox_to_anchor=(1.01, 1), loc="upper left", ncol=1)

    fig.tight_layout()
    return fig, ax


#------------------------------------------

#-------------------------------------------

def plot_dccd_chi2_vs_time_by_target_filter(
    df,
    filter_col="FILTER",
    filter_select=None,
    time_col="Time",
    target_col="TARGET",
    dccd_col="D_CCD [mm]",
    chi2_col="CHI2_FIT",

    # bornes / seuils
    dccd_min_fig=None,
    dccd_max_fig=None,
    dccd_min_cut=None,
    dccd_max_cut=None,
    chi2_min_fig=None,
    chi2_max_fig=None,
    chi2_cut=None,


    # style
    marker="+",
    lw=5,
    alpha=0.5,
    date_format="%y-%m-%d",
    suptitle=None,

    # affichage
    per_target=False,  # si True, une paire de plot par TARGET
    force_global_time_xlim=True,
    axs=None,
    figsize=(18, 8),
    tag=None,
):
    """
    Plot DCCD vs time et CHI2 vs time pour un filtre donné, avec couleur par TARGET.
    - per_target=False: toutes les TARGET sur une seule paire de plots
    - per_target=True: une paire de plots par TARGET (axes empilés verticalement)
    """

    data = df.copy()

    # ----------------------------
    # Filtrer le dataframe si filter_select fourni
    # ----------------------------
    if filter_select is not None:
        data = data[data[filter_col] == filter_select]


    # ----------------------------
    # Range temporel global (pour xlim)
    # ----------------------------
    time_min = data[time_col].min()
    time_max = data[time_col].max()

    targets = data[target_col].unique()
    n_targets = len(targets)

    # ----------------------------
    # Palette dynamique TARGET
    # ----------------------------
    if n_targets <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("hsv")
    target_palette = {t: mcolors.to_hex(cmap(i / n_targets)) for i, t in enumerate(targets)}

    date_form = DateFormatter(date_format)

    # ----------------------------
    # Cas per_target = False (tout sur une paire de plots)
    # ----------------------------
    if not per_target:
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(figsize[0], figsize[1]),constrained_layout=True)
        else:
            fig = axs[0].figure
        ax_dccd, ax_chi2 = axs

        # DCCD vs Time
        for t in targets:
            sub = data[data[target_col] == t]
            ax_dccd.scatter(
                sub[time_col],
                sub[dccd_col],
                color=target_palette[t],
                marker=marker,
                lw=lw,
                alpha=alpha,
                label=t
            )

        if dccd_min_cut is not None:
            ax_dccd.axhline(dccd_min_cut, ls="-.", c="k")
        if dccd_max_cut is not None:
            ax_dccd.axhline(dccd_max_cut, ls="-.", c="k")


        ylim_min = dccd_min_fig if dccd_min_fig is not None else dccd_min_cut
        ylim_max = dccd_max_fig if dccd_max_fig is not None else dccd_max_cut
        if ylim_min is not None and ylim_max is not None:
            ax_dccd.set_ylim(ylim_min, ylim_max)


        ax_dccd.set_ylabel("D_CCD [mm]")
        ax_dccd.set_xlabel("Time")
        ax_dccd.set_title(f"DCCD vs Time – Filter: {filter_select}")
        ax_dccd.xaxis.set_major_formatter(date_form)
        ax_dccd.grid(True, alpha=0.3)
        plt.setp(ax_dccd.get_xticklabels(), rotation=45, ha="right")

        # CHI2 vs Time
        for t in targets:
            sub = data[data[target_col] == t]
            ax_chi2.scatter(
                sub[time_col],
                sub[chi2_col],
                color=target_palette[t],
                marker=marker,
                lw=lw,
                alpha=alpha,
                label=t
            )
        ax_chi2.set_yscale("log")
        if chi2_cut is not None:
            ax_chi2.axhline(chi2_cut, ls="-.", c="k")


        ylim_min = chi2_min_fig if chi2_min_fig is not None else 1.
        ylim_max = chi2_max_fig if chi2_max_fig is not None else chi2_cut
        if ylim_min is not None and ylim_max is not None:
            ax_chi2.set_ylim(ylim_min, ylim_max)


        ax_chi2.set_ylabel("CHI2_FIT")
        ax_chi2.set_xlabel("Time")
        ax_chi2.set_title(f"CHI2 vs Time – Filter: {filter_select}")
        ax_chi2.xaxis.set_major_formatter(date_form)
        ax_chi2.grid(True, alpha=0.3)
        plt.setp(ax_chi2.get_xticklabels(), rotation=45, ha="right")

        # légende unique à gauche
        handles = [plt.Line2D([0], [0], marker=marker, color=target_palette[t], linestyle="", markersize=8) for t in targets]
        fig.legend(handles, targets, title="TARGET", loc="center left", bbox_to_anchor=(1.01, 0.55), ncol=2)

        if suptitle:
            if tag is not None:
                suptitle += " "
                suptitle += tag
            fig.suptitle(suptitle,fontsize=16)
        #fig.tight_layout(rect=[0.05, 0, 1, 1])  # espace pour légende

    # ----------------------------
    # Cas per_target = True (une paire de plot par TARGET)
    # ----------------------------
    else:
        n_panels = n_targets
        if axs is None:
            fig, axs = plt.subplots(n_panels, 2, figsize=(figsize[0], figsize[1]*n_panels),constrained_layout=True)
            if n_panels == 1:
                axs = [axs]
        else:
            fig = axs[0].figure

        for i, t in enumerate(targets):
            ax_dccd, ax_chi2 = axs[i]

            sub = data[data[target_col] == t]

            # DCCD
            ax_dccd.scatter(
                sub[time_col],
                sub[dccd_col],
                color=target_palette[t],
                marker=marker,
                lw=lw,
                alpha=alpha,
                label=t
            )
            if force_global_time_xlim:
                ax_dccd.set_xlim(time_min, time_max)
            if dccd_min_fig is not None and dccd_max_fig is not None:
                ax_dccd.set_ylim(dccd_min_fig, dccd_max_fig)
            if dccd_min_cut is not None:
                ax_dccd.axhline(dccd_min_cut, ls="-.", c="k")
            if dccd_max_cut is not None:
                ax_dccd.axhline(dccd_max_cut, ls="-.", c="k")



            ax_dccd.set_ylabel("D_CCD [mm]")
            ax_dccd.set_xlabel("Time")
            ax_dccd.set_title(f"{t} – DCCD vs Time – Filter: {filter_select}")
            ax_dccd.xaxis.set_major_formatter(date_form)
            ax_dccd.grid(True, alpha=0.3)
            plt.setp(ax_dccd.get_xticklabels(), rotation=45, ha="right")

            # CHI2
            ax_chi2.scatter(
                sub[time_col],
                sub[chi2_col],
                color=target_palette[t],
                marker=marker,
                lw=lw,
                alpha=alpha,
                label=t
            )



            ax_chi2.set_yscale("log")
            if chi2_cut is not None:
                ax_chi2.axhline(chi2_cut, ls="-.", c="k")

            if chi2_min_fig is not None and chi2_max_fig is not None:
                ax_chi2.set_ylim(chi2_min_fig, chi2_max_fig)


            if force_global_time_xlim:
                ax_chi2.set_xlim(time_min, time_max)

            ax_chi2.set_ylabel("CHI2_FIT")
            ax_chi2.set_xlabel("Time")
            ax_chi2.set_title(f"{t} – CHI2 vs Time – Filter: {filter_select}")
            ax_chi2.xaxis.set_major_formatter(date_form)
            ax_chi2.grid(True, alpha=0.3)
            plt.setp(ax_chi2.get_xticklabels(), rotation=45, ha="right")

        if suptitle:
            fig.suptitle(suptitle,fontsize=16)
        #fig.tight_layout()


    # pour avoir les legendes
    # supprimer les légendes locales
    #for ax in axs.flat():
    #    leg = ax.get_legend()
    #    if leg is not None:
    #        leg.remove()

    # légende globale
    #handles, labels = axs[0].get_legend_handles_labels()

    #fig.legend(
    #    handles,
    #    labels,
    #    loc="center left",
    #    bbox_to_anchor=(1.01, 0.55),
    #    title="Target"
    #)

    return fig, axs


#------------------------------------------
#
#------------------------------------------
def plot_dccd_chi2_histo_by_target_filter(
    df,
    filter_col="FILTER",
    filter_select=None,
    target_col="TARGET",
    dccd_col="D_CCD [mm]",
    chi2_col="CHI2_FIT",

    # bornes / seuils
    dccd_min_fig=None,
    dccd_max_fig=None,
    dccd_min_cut=None,
    dccd_max_cut=None,
    chi2_min_fig=None,
    chi2_max_fig=None,
    chi2_cut=None,

    # histogramme
    bins_dccd=100,
    bins_chi2=100,
    density=False,

    # style
    lw=4,

    suptitle=None,

    # affichage
    per_target=False,
    axs=None,
    figsize=(18, 8),
    tag=None,


):
    """
    Histogrammes DCCD et CHI2 pour un filtre donné.
    - per_target=False : tous les TARGET superposés sur une paire de plots
    - per_target=True  : une paire de plots par TARGET
    """

    data = df.copy()

    # ----------------------------
    # Filtrage par filtre
    # ----------------------------
    if filter_select is not None:
        data = data[data[filter_col] == filter_select]

    targets = np.sort(data[target_col].unique())
    n_targets = len(targets)



    # ----------------------------
    # Palette dynamique TARGET
    # ----------------------------
    if n_targets <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("hsv")
    target_palette = {t: mcolors.to_hex(cmap(i / n_targets)) for i, t in enumerate(targets)}


    # ----------------------------
    # Définition des bins communs
    # ----------------------------
    dccd_vals = data[dccd_col].dropna()
    chi2_vals = data[chi2_col].dropna()

    dccd_bins = np.linspace(
        dccd_vals.min() if dccd_min_fig is None else dccd_min_fig,
        dccd_vals.max() if dccd_max_fig is None else dccd_max_fig,
        bins_dccd,
    )

    chi2_bins = np.logspace(
        np.log10(chi2_vals[chi2_vals > 0].min()),
        np.log10(chi2_vals.max()),
        bins_chi2,
    )

    # ============================================================
    # Cas per_target = False
    # ============================================================
    if not per_target:
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
        else:
            fig = axs[0].figure

        ax_dccd, ax_chi2 = axs

        # --- DCCD histogrammes
        for t in targets:
            sub = data[data[target_col] == t]
            ax_dccd.hist(
                sub[dccd_col].dropna(),
                bins=dccd_bins,
                histtype="step",
                lw=lw,
                density=density,
                color=target_palette[t],
                label=t,
            )

        if dccd_min_cut is not None:
            ax_dccd.axvline(dccd_min_cut, ls="-.", c="k")
        if dccd_max_cut is not None:
            ax_dccd.axvline(dccd_max_cut, ls="-.", c="k")

        if dccd_min_fig is not None and dccd_max_fig is not None:
            ax_dccd.set_xlim(dccd_min_fig, dccd_max_fig)

        ax_dccd.set_xlabel("D_CCD [mm]")
        ax_dccd.set_ylabel("Density" if density else "Counts")
        ax_dccd.set_title(f"DCCD histogram – Filter: {filter_select}")
        ax_dccd.grid(True, alpha=0.3)

        # --- CHI2 histogrammes
        for t in targets:
            sub = data[data[target_col] == t]
            ax_chi2.hist(
                sub[chi2_col].dropna(),
                bins=chi2_bins,
                histtype="step",
                lw=lw,
                density=density,
                color=target_palette[t],
                label=t,
            )

        ax_chi2.set_xscale("log")

        if chi2_cut is not None:
            ax_chi2.axvline(chi2_cut, ls="-.", c="k")

        if chi2_min_fig is not None and chi2_max_fig is not None:
            ax_chi2.set_xlim(chi2_min_fig, chi2_max_fig)

        ax_chi2.set_xlabel("CHI2_FIT")
        ax_chi2.set_ylabel("Density" if density else "Counts")
        ax_chi2.set_title(f"CHI2 histogram – Filter: {filter_select}")
        ax_chi2.grid(True, alpha=0.3)

        # --- légende globale
        handles = [
            plt.Line2D([0], [0], color=target_palette[t], lw=lw)
            for t in targets
        ]

        fig.legend(
            handles,
            targets,
            title="TARGET",
            loc="center left",
            bbox_to_anchor=(1.01, 0.55),
            ncol=2,
        )

        if suptitle:
            if tag is not None:
                suptitle = f"{suptitle} {tag}"
            fig.suptitle(suptitle, fontsize=16)

    # ============================================================
    # Cas per_target = True
    # ============================================================
    else:
        n_panels = n_targets

        if axs is None:
            fig, axs = plt.subplots(
                n_panels, 2,
                figsize=(figsize[0], figsize[1] * n_panels),
                constrained_layout=True,
            )
            if n_panels == 1:
                axs = [axs]
        else:
            fig = axs[0].figure

        for i, t in enumerate(targets):
            ax_dccd, ax_chi2 = axs[i]
            sub = data[data[target_col] == t]

            # --- DCCD
            ax_dccd.hist(
                sub[dccd_col].dropna(),
                bins=dccd_bins,
                histtype="step",
                lw=lw,
                density=density,
                color=target_palette[t],
            )

            if dccd_min_cut is not None:
                ax_dccd.axvline(dccd_min_cut, ls="-.", c="k")
            if dccd_max_cut is not None:
                ax_dccd.axvline(dccd_max_cut, ls="-.", c="k")

            if dccd_min_fig is not None and dccd_max_fig is not None:
                ax_dccd.set_xlim(dccd_min_fig, dccd_max_fig)

            ax_dccd.set_title(f"{t} – DCCD – Filter: {filter_select}")
            ax_dccd.set_xlabel("D_CCD [mm]")
            ax_dccd.set_ylabel("Density" if density else "Counts")
            ax_dccd.grid(True, alpha=0.3)

            # --- CHI2
            ax_chi2.hist(
                sub[chi2_col].dropna(),
                bins=chi2_bins,
                histtype="step",
                lw=lw,
                density=density,
                color=target_palette[t],
            )

            ax_chi2.set_xscale("log")

            if chi2_cut is not None:
                ax_chi2.axvline(chi2_cut, ls="-.", c="k")

            if chi2_min_fig is not None and chi2_max_fig is not None:
                ax_chi2.set_xlim(chi2_min_fig, chi2_max_fig)

            ax_chi2.set_title(f"{t} – CHI2 – Filter: {filter_select}")
            ax_chi2.set_xlabel("CHI2_FIT")
            ax_chi2.set_ylabel("Density" if density else "Counts")
            ax_chi2.grid(True, alpha=0.3)

        if suptitle:
            fig.suptitle(suptitle, fontsize=16)

    return fig, axs


#------------------------------------------
#
#--------------------------------------------------
def plot_dccd_chi2_vs_time_by_target_filter_colorsedtype(
    df,
    filter_col="FILTER",
    filter_select=None,
    time_col="Time",
    target_col="TARGET",
    dccd_col="D_CCD [mm]",
    chi2_col="CHI2_FIT",

    # bornes / seuils
    dccd_min_fig=None,
    dccd_max_fig=None,
    dccd_min_cut=None,
    dccd_max_cut=None,
    chi2_min_fig=None,
    chi2_max_fig=None,
    chi2_cut=None,


    # style
    marker="+",
    lw=5,
    alpha=0.5,
    date_format="%y-%m-%d",
    suptitle=None,

    # affichage
    per_target=False,  # si True, une paire de plot par TARGET
    force_global_time_xlim=True,
    axs=None,
    figsize=(18, 8),
    tag=None,

    # NEW
    target_palette=None,   # dict: TARGET -> color
):
    """
    Plot DCCD vs time et CHI2 vs time pour un filtre donné, avec couleur par TARGET.
    - per_target=False: toutes les TARGET sur une seule paire de plots
    - per_target=True: une paire de plots par TARGET (axes empilés verticalement)
    """


    data = df.copy()



    # ----------------------------
    # Palette TARGET (external)
    # ----------------------------
    targets = data[target_col].unique()

    #if target_palette is None:
    #    raise ValueError(
    #    "target_palette must be provided as a dict: {target: color}"
    #    )
    if target_palette is None:
        target_palette = {}

    missing = set(targets) - set(target_palette.keys())
    if missing:
        #raise ValueError(
        pprint(
            f"Missing colors for targets: {missing}"
        )

    # Ensure every target has a color
    effective_palette = {
        t: target_palette.get(t, DEFAULT_TARGET_COLOR) for t in targets}

    # ----------------------------
    # Filtrer le dataframe si filter_select fourni
    # ----------------------------
    if filter_select is not None:
        data = data[data[filter_col] == filter_select]



    # ----------------------------
    # Range temporel global (pour xlim)
    # ----------------------------
    time_min = data[time_col].min()
    time_max = data[time_col].max()

    targets = data[target_col].unique()
    n_targets = len(targets)

    # ----------------------------
    # Palette dynamique TARGET
    # ----------------------------

    if target_palette is None:
        if n_targets <= 20:
            cmap = plt.get_cmap("tab20")
        else:
            cmap = plt.get_cmap("hsv")

        target_palette = {t: mcolors.to_hex(cmap(i / n_targets)) for i, t in enumerate(targets)}

    date_form = DateFormatter(date_format)

    # ----------------------------
    # Cas per_target = False (tout sur une paire de plots)
    # ----------------------------
    if not per_target:
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(figsize[0], figsize[1]),constrained_layout=True)
        else:
            fig = axs[0].figure
        ax_dccd, ax_chi2 = axs

        # DCCD vs Time
        for t in targets:
            sub = data[data[target_col] == t]
            ax_dccd.scatter(
                sub[time_col],
                sub[dccd_col],
                color=effective_palette[t],
                marker=marker,
                lw=lw,
                alpha=alpha,
                label=t
            )

        if dccd_min_cut is not None:
            ax_dccd.axhline(dccd_min_cut, ls="-.", c="k")
        if dccd_max_cut is not None:
            ax_dccd.axhline(dccd_max_cut, ls="-.", c="k")


        ylim_min = dccd_min_fig if dccd_min_fig is not None else dccd_min_cut
        ylim_max = dccd_max_fig if dccd_max_fig is not None else dccd_max_cut
        if ylim_min is not None and ylim_max is not None:
            ax_dccd.set_ylim(ylim_min, ylim_max)


        ax_dccd.set_ylabel("D_CCD [mm]")
        ax_dccd.set_xlabel("Time")
        ax_dccd.set_title(f"DCCD vs Time – Filter: {filter_select}")
        ax_dccd.xaxis.set_major_formatter(date_form)
        ax_dccd.grid(True, alpha=0.3)
        plt.setp(ax_dccd.get_xticklabels(), rotation=45, ha="right")

        # CHI2 vs Time
        for t in targets:
            sub = data[data[target_col] == t]
            ax_chi2.scatter(
                sub[time_col],
                sub[chi2_col],
                color=effective_palette[t],
                marker=marker,
                lw=lw,
                alpha=alpha,
                label=t
            )
        ax_chi2.set_yscale("log")
        if chi2_cut is not None:
            ax_chi2.axhline(chi2_cut, ls="-.", c="k")


        ylim_min = chi2_min_fig if chi2_min_fig is not None else 1.
        ylim_max = chi2_max_fig if chi2_max_fig is not None else chi2_cut
        if ylim_min is not None and ylim_max is not None:
            ax_chi2.set_ylim(ylim_min, ylim_max)


        ax_chi2.set_ylabel("CHI2_FIT")
        ax_chi2.set_xlabel("Time")
        ax_chi2.set_title(f"CHI2 vs Time – Filter: {filter_select}")
        ax_chi2.xaxis.set_major_formatter(date_form)
        ax_chi2.grid(True, alpha=0.3)
        plt.setp(ax_chi2.get_xticklabels(), rotation=45, ha="right")

        # légende unique à gauche
        handles = [plt.Line2D([0], [0], marker=marker, color=effective_palette[t], linestyle="", markersize=8) for t in targets]
        fig.legend(handles, targets, title="TARGET", loc="center left", bbox_to_anchor=(1.01, 0.55), ncol=2)

        if suptitle:
            if tag is not None:
                suptitle += " "
                suptitle += tag
            fig.suptitle(suptitle,fontsize=16)
        #fig.tight_layout(rect=[0.05, 0, 1, 1])  # espace pour légende

    # ----------------------------
    # Cas per_target = True (une paire de plot par TARGET)
    # ----------------------------
    else:
        n_panels = n_targets
        if axs is None:
            fig, axs = plt.subplots(n_panels, 2, figsize=(figsize[0], figsize[1]*n_panels),constrained_layout=True)
            if n_panels == 1:
                axs = [axs]
        else:
            fig = axs[0].figure

        for i, t in enumerate(targets):
            ax_dccd, ax_chi2 = axs[i]

            sub = data[data[target_col] == t]

            # DCCD
            ax_dccd.scatter(
                sub[time_col],
                sub[dccd_col],
                color=effective_palette[t],
                marker=marker,
                lw=lw,
                alpha=alpha,
                label=t
            )

            if force_global_time_xlim:
                ax_dccd.set_xlim(time_min, time_max)

            if dccd_min_fig is not None and dccd_max_fig is not None:
                ax_dccd.set_ylim(dccd_min_fig, dccd_max_fig)
            if dccd_min_cut is not None:
                ax_dccd.axhline(dccd_min_cut, ls="-.", c="k")
            if dccd_max_cut is not None:
                ax_dccd.axhline(dccd_max_cut, ls="-.", c="k")



            ax_dccd.set_ylabel("D_CCD [mm]")
            ax_dccd.set_xlabel("Time")
            ax_dccd.set_title(f"{t} – DCCD vs Time – Filter: {filter_select}")
            ax_dccd.xaxis.set_major_formatter(date_form)
            ax_dccd.grid(True, alpha=0.3)
            plt.setp(ax_dccd.get_xticklabels(), rotation=45, ha="right")

            # CHI2
            ax_chi2.scatter(
                sub[time_col],
                sub[chi2_col],
                color=effective_palette[t],
                marker=marker,
                lw=lw,
                alpha=alpha,
                label=t
            )

            if force_global_time_xlim:
                ax_chi2.set_xlim(time_min, time_max)

            ax_chi2.set_yscale("log")
            if chi2_cut is not None:
                ax_chi2.axhline(chi2_cut, ls="-.", c="k")

            if chi2_min_fig is not None and chi2_max_fig is not None:
                ax_chi2.set_ylim(chi2_min_fig, chi2_max_fig)

            ax_chi2.set_ylabel("CHI2_FIT")
            ax_chi2.set_xlabel("Time")
            ax_chi2.set_title(f"{t} – CHI2 vs Time – Filter: {filter_select}")
            ax_chi2.xaxis.set_major_formatter(date_form)
            ax_chi2.grid(True, alpha=0.3)
            plt.setp(ax_chi2.get_xticklabels(), rotation=45, ha="right")

        if suptitle:
            fig.suptitle(suptitle,fontsize=16)
        #fig.tight_layout()


    # pour avoir les legendes
    # supprimer les légendes locales
    #for ax in axs.flat():
    #    leg = ax.get_legend()
    #    if leg is not None:
    #        leg.remove()

    # légende globale
    #handles, labels = axs[0].get_legend_handles_labels()

    #fig.legend(
    #    handles,
    #    labels,
    #    loc="center left",
    #    bbox_to_anchor=(1.01, 0.55),
    #    title="Target"
    #)

    return fig, axs
#------------------------

#------------------------
def plot_dccd_chi2_histo_by_target_filter_colorsedtype(
    df,
    filter_col="FILTER",
    filter_select=None,
    target_col="TARGET",
    dccd_col="D_CCD [mm]",
    chi2_col="CHI2_FIT",

    # bornes / seuils
    dccd_min_fig=None,
    dccd_max_fig=None,
    dccd_min_cut=None,
    dccd_max_cut=None,
    chi2_min_fig=None,
    chi2_max_fig=None,
    chi2_cut=None,

    # histogramme
    bins_dccd=100,
    bins_chi2=100,
    density=False,

    # style
    lw=4,

    suptitle=None,

    # affichage
    per_target=False,
    axs=None,
    figsize=(18, 8),
    tag=None,

# colors
    target_palette=None,   # dict: TARGET -> color

):
    """
    Histogrammes DCCD et CHI2 pour un filtre donné.
    - per_target=False : tous les TARGET superposés sur une paire de plots
    - per_target=True  : une paire de plots par TARGET
    """

    data = df.copy()

    # ----------------------------
    # Filter data ccorording the requested filter
    # ----------------------------
    if filter_select is not None:
        data = data[data[filter_col] == filter_select]

    targets = np.sort(data[target_col].unique())
    n_targets = len(targets)


    # ----------------------------
    # If no palette is provided a color palette is generated otherwise
    # use  the one provided but check for missing targets
    # ----------------------------
    if target_palette is None:
        if n_targets <= 20:
            cmap = plt.get_cmap("tab20")
        else:
            cmap = plt.get_cmap("hsv")
        target_palette = {t: mcolors.to_hex(cmap(i / n_targets)) for i, t in enumerate(targets)}
    else:
        # so here the target palette exist but de fefault color is attributed for missing targets
        missing = set(targets) - set(target_palette.keys())
        if missing:
            print(f"Missing colors for targets → defaulting to black: {missing}")

        effective_palette = {t: target_palette.get(t, DEFAULT_TARGET_COLOR) for t in targets}
        target_palette = effective_palette
    # ----------------------------
    # Définition des bins communs
    # ----------------------------
    dccd_vals = data[dccd_col].dropna()
    chi2_vals = data[chi2_col].dropna()

    dccd_bins = np.linspace(
        dccd_vals.min() if dccd_min_fig is None else dccd_min_fig,
        dccd_vals.max() if dccd_max_fig is None else dccd_max_fig,
        bins_dccd,
    )

    chi2_bins = np.logspace(
        np.log10(chi2_vals[chi2_vals > 0].min()),
        np.log10(chi2_vals.max()),
        bins_chi2,
    )

    # ============================================================
    # Cas per_target = False
    # ============================================================
    if not per_target:
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
        else:
            fig = axs[0].figure

        ax_dccd, ax_chi2 = axs

        # --- DCCD histogrammes
        for t in targets:
            sub = data[data[target_col] == t]
            ax_dccd.hist(
                sub[dccd_col].dropna(),
                bins=dccd_bins,
                histtype="step",
                lw=lw,
                density=density,
                color=target_palette[t],
                label=t,
            )

        if dccd_min_cut is not None:
            ax_dccd.axvline(dccd_min_cut, ls="-.", c="k")
        if dccd_max_cut is not None:
            ax_dccd.axvline(dccd_max_cut, ls="-.", c="k")

        if dccd_min_fig is not None and dccd_max_fig is not None:
            ax_dccd.set_xlim(dccd_min_fig, dccd_max_fig)

        ax_dccd.set_xlabel("D_CCD [mm]")
        ax_dccd.set_ylabel("Density" if density else "Counts")
        ax_dccd.set_title(f"DCCD histogram – Filter: {filter_select}")
        ax_dccd.grid(True, alpha=0.3)

        # --- CHI2 histogrammes
        for t in targets:
            sub = data[data[target_col] == t]
            ax_chi2.hist(
                sub[chi2_col].dropna(),
                bins=chi2_bins,
                histtype="step",
                lw=lw,
                density=density,
                color=target_palette[t],
                label=t,
            )

        ax_chi2.set_xscale("log")

        if chi2_cut is not None:
            ax_chi2.axvline(chi2_cut, ls="-.", c="k")

        if chi2_min_fig is not None and chi2_max_fig is not None:
            ax_chi2.set_xlim(chi2_min_fig, chi2_max_fig)

        ax_chi2.set_xlabel("CHI2_FIT")
        ax_chi2.set_ylabel("Density" if density else "Counts")
        ax_chi2.set_title(f"CHI2 histogram – Filter: {filter_select}")
        ax_chi2.grid(True, alpha=0.3)

        # --- légende globale
        handles = [
            plt.Line2D([0], [0], color=target_palette[t], lw=lw)
            for t in targets
        ]

        fig.legend(
            handles,
            targets,
            title="TARGET",
            loc="center left",
            bbox_to_anchor=(1.01, 0.55),
            ncol=2,
        )

        if suptitle:
            if tag is not None:
                suptitle = f"{suptitle} {tag}"
            fig.suptitle(suptitle, fontsize=16)

    # ============================================================
    # Cas per_target = True
    # ============================================================
    else:
        n_panels = n_targets

        if axs is None:
            fig, axs = plt.subplots(
                n_panels, 2,
                figsize=(figsize[0], figsize[1] * n_panels),
                constrained_layout=True,
            )
            if n_panels == 1:
                axs = [axs]
        else:
            fig = axs[0].figure

        for i, t in enumerate(targets):
            ax_dccd, ax_chi2 = axs[i]
            sub = data[data[target_col] == t]

            # --- DCCD
            ax_dccd.hist(
                sub[dccd_col].dropna(),
                bins=dccd_bins,
                histtype="step",
                lw=lw,
                density=density,
                color=target_palette[t],
            )

            if dccd_min_cut is not None:
                ax_dccd.axvline(dccd_min_cut, ls="-.", c="k")
            if dccd_max_cut is not None:
                ax_dccd.axvline(dccd_max_cut, ls="-.", c="k")

            if dccd_min_fig is not None and dccd_max_fig is not None:
                ax_dccd.set_xlim(dccd_min_fig, dccd_max_fig)

            ax_dccd.set_title(f"{t} – DCCD – Filter: {filter_select}")
            ax_dccd.set_xlabel("D_CCD [mm]")
            ax_dccd.set_ylabel("Density" if density else "Counts")
            ax_dccd.grid(True, alpha=0.3)

            # --- CHI2
            ax_chi2.hist(
                sub[chi2_col].dropna(),
                bins=chi2_bins,
                histtype="step",
                lw=lw,
                density=density,
                color=target_palette[t],
            )

            ax_chi2.set_xscale("log")

            if chi2_cut is not None:
                ax_chi2.axvline(chi2_cut, ls="-.", c="k")

            if chi2_min_fig is not None and chi2_max_fig is not None:
                ax_chi2.set_xlim(chi2_min_fig, chi2_max_fig)

            ax_chi2.set_title(f"{t} – CHI2 – Filter: {filter_select}")
            ax_chi2.set_xlabel("CHI2_FIT")
            ax_chi2.set_ylabel("Density" if density else "Counts")
            ax_chi2.grid(True, alpha=0.3)

        if suptitle:
            fig.suptitle(suptitle, fontsize=16)

    return fig, axs


#------------------------

#------------------------
def plot_dccd_chi2_histo_by_target_filter_colorsedtype_bad(
    df,
    filter_col="FILTER",
    filter_select=None,
    target_col="TARGET",
    dccd_col="D_CCD [mm]",
    chi2_col="CHI2_FIT",

    # histogram control
    bins_dccd=100,
    bins_chi2=100,
    density=False,

    # bornes / seuils (FIG = axes limits, CUT = vertical lines)
    dccd_min_fig=None,
    dccd_max_fig=None,
    dccd_min_cut=None,
    dccd_max_cut=None,
    chi2_min_fig=None,
    chi2_max_fig=None,
    chi2_cut=None,

    # style
    lw=4,
    alpha=0.9,
    suptitle=None,

    # affichage
    per_target=False,
    axs=None,
    figsize=(18, 8),
    tag=None,

    # colors
    target_palette=None,   # dict: TARGET -> color
):
    """
    Histogrammes DCCD et CHI2 pour un filtre donné.
    - 1 couleur par TARGET
    - per_target=False : tous les TARGET superposés
    - per_target=True  : une paire de plots par TARGET
    """

    data = df.copy()



    dccd_range = (
    dccd_min_fig if dccd_min_fig is not None else None,
    dccd_max_fig if dccd_max_fig is not None else None,
)

    chi2_range = (
    chi2_min_fig if chi2_min_fig is not None else None,
    chi2_max_fig if chi2_max_fig is not None else None,
)


    # ----------------------------
    # Filtre
    # ----------------------------
    if filter_select is not None:
        data = data[data[filter_col] == filter_select]

    targets = data[target_col].unique()
    n_targets = len(targets)

    # ----------------------------
    # Palette TARGET
    # ----------------------------
    if target_palette is None:
        target_palette = {}

    missing = set(targets) - set(target_palette.keys())
    if missing:
        pprint(f"Missing colors for targets → defaulting to black: {missing}")

    effective_palette = {
        t: target_palette.get(t, DEFAULT_TARGET_COLOR)
        for t in targets
    }

    # ============================================================
    # CASE per_target = False
    # ============================================================
    if not per_target:
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
        else:
            fig = axs[0].figure

        ax_dccd, ax_chi2 = axs

        # ---------- DCCD histogram ----------
        for t in targets:
            sub = data[data[target_col] == t]
            ax_dccd.hist(
                sub[dccd_col],
                bins=bins_dccd,
                range=dccd_range,
                histtype="step",
                lw=lw,
                alpha=alpha,
                color=effective_palette[t],
                label=t,
                density=density,
            )

        if dccd_min_cut is not None:
            ax_dccd.axvline(dccd_min_cut, ls="-.", c="k")
        if dccd_max_cut is not None:
            ax_dccd.axvline(dccd_max_cut, ls="-.", c="k")

        if dccd_min_fig is not None or dccd_max_fig is not None:
            ax_dccd.set_xlim(dccd_min_fig, dccd_max_fig)

        ax_dccd.set_xlabel("D_CCD [mm]")
        ax_dccd.set_ylabel("Counts" if not density else "Density")
        ax_dccd.set_title(f"DCCD histogram – Filter: {filter_select}")
        ax_dccd.grid(True, alpha=0.3)

        # ---------- CHI2 histogram ----------
        for t in targets:
            sub = data[data[target_col] == t]
            ax_chi2.hist(
                sub[chi2_col],
                bins=bins_chi2,
                range=chi2_range,
                histtype="step",
                lw=lw,
                alpha=alpha,
                color=effective_palette[t],
                label=t,
                density=density,
            )

        ax_chi2.set_xscale("log")

        if chi2_cut is not None:
            ax_chi2.axvline(chi2_cut, ls="-.", c="k")

        if chi2_min_fig is not None or chi2_max_fig is not None:
            ax_chi2.set_xlim(chi2_min_fig, chi2_max_fig)

        ax_chi2.set_xlabel("CHI2_FIT")
        ax_chi2.set_ylabel("Counts" if not density else "Density")
        ax_chi2.set_title(f"CHI2 histogram – Filter: {filter_select}")
        ax_chi2.grid(True, alpha=0.3)

        # ---------- Global legend ----------
        handles = [
            plt.Line2D([0], [0], color=effective_palette[t], lw=lw)
            for t in targets
        ]
        fig.legend(
            handles,
            targets,
            title="TARGET",
            loc="center left",
            bbox_to_anchor=(1.01, 0.55),
            ncol=2,
        )

        if suptitle:
            if tag is not None:
                suptitle = f"{suptitle} {tag}"
            fig.suptitle(suptitle, fontsize=16)

    # ============================================================
    # CASE per_target = True
    # ============================================================
    else:
        if axs is None:
            fig, axs = plt.subplots(
                n_targets, 2,
                figsize=(figsize[0], figsize[1] * n_targets),
                constrained_layout=True,
            )
            if n_targets == 1:
                axs = [axs]
        else:
            fig = axs[0].figure

        for i, t in enumerate(targets):
            ax_dccd, ax_chi2 = axs[i]
            sub = data[data[target_col] == t]

            # DCCD
            ax_dccd.hist(
                sub[dccd_col],
                bins=bins_dccd,
                histtype="step",
                lw=lw,
                alpha=alpha,
                color=effective_palette[t],
                density=density,
            )
            if dccd_min_cut is not None:
                ax_dccd.axvline(dccd_min_cut, ls="-.", c="k")
            if dccd_max_cut is not None:
                ax_dccd.axvline(dccd_max_cut, ls="-.", c="k")
            if dccd_min_fig is not None or dccd_max_fig is not None:
                ax_dccd.set_xlim(dccd_min_fig, dccd_max_fig)

            ax_dccd.set_title(f"{t} – DCCD")
            ax_dccd.set_xlabel("D_CCD [mm]")
            ax_dccd.set_ylabel("Counts" if not density else "Density")
            ax_dccd.grid(True, alpha=0.3)

            # CHI2
            ax_chi2.hist(
                sub[chi2_col],
                bins=bins_chi2,
                histtype="step",
                lw=lw,
                alpha=alpha,
                color=effective_palette[t],
                density=density,
            )
            ax_chi2.set_xscale("log")
            if chi2_cut is not None:
                ax_chi2.axvline(chi2_cut, ls="-.", c="k")
            if chi2_min_fig is not None or chi2_max_fig is not None:
                ax_chi2.set_xlim(chi2_min_fig, chi2_max_fig)

            ax_chi2.set_title(f"{t} – CHI2")
            ax_chi2.set_xlabel("CHI2_FIT")
            ax_chi2.set_ylabel("Counts" if not density else "Density")
            ax_chi2.grid(True, alpha=0.3)

        if suptitle:
            fig.suptitle(suptitle, fontsize=16)

    return fig, axs


#------------------------
def summarize_dccd_chi2(df, target_col="TARGET", filter_col="FILTER",
                        dccd_col="D_CCD [mm]", chi2_col="CHI2_FIT"):
    """
    Summarize DCCD and CHI2 for each TARGET and FILTER.

    Returns a pandas DataFrame with:
        TARGET | FILTER | mean_DCCD | sigma_DCCD | mean_CHI2 | sigma_CHI2
    """
    summary = (
        df
        .groupby([target_col, filter_col])
        .agg(
            N=("CHI2_FIT", "size"), 
            mean_DCCD = (dccd_col, "mean"),
            sigma_DCCD = (dccd_col, "std"),
            mean_CHI2 = (chi2_col, "mean"),
            sigma_CHI2 = (chi2_col, "std")
        )
        .reset_index()
    )
    return summary

#------------------------------------------------------------------
# What induces bad chi2
#-------------------------------------------------------------------
def plot_params_and_chi2_vs_time(
    df,
    time_col,
    filter_col,
    chi2_col,
    params,                      # list of parameter column names

    # display
    marker="+",
    lw=5,
    alpha=0.5,

    # datetime
    date_format="%y-%m-%d",

    # scaling
    chi2_log=True,

    # optional bounds
    param_ylim=None,              # dict: {param: (ymin, ymax)}
    chi2_cut=None,

    #colors of params   
    param_colors=None,             # dict: {param: color}

    # titles
    panel_titles=None,            # dict: {param: title}
    suptitle=None,

    # axes handling
    axs=None,
    figsize=(18, 4),
):
    """
    Plot an arbitrary list of parameters vs time, each compared with CHI2_FIT
    using a twin y-axis.

    Parameters
    ----------
    params : list of str
        List of dataframe column names to plot vs time.
    param_ylim : dict, optional
        {param: (ymin, ymax)} for left axes.
    panel_titles : dict, optional
        {param: title} for panel titles.
    axs : list of matplotlib.axes.Axes, optional
        External axes (length must match len(params)).
    """


    data = df.copy()

    # ----------------------------
    # Datetime handling
    # ----------------------------
    if is_datetime64_any_dtype(data[time_col]):
        try:
            data[time_col] = data[time_col].dt.tz_convert(None)
        except TypeError:
            pass

    n_panels = len(params)

    # ----------------------------
    # Axes creation
    # ----------------------------
    if axs is None:
        fig, axs = plt.subplots(
            n_panels, 1,
            figsize=(figsize[0], figsize[1] * n_panels),
            sharex=False,
            constrained_layout=True,
        )
        if n_panels == 1:
            axs = [axs]
    else:
        fig = axs[0].figure

    date_form = DateFormatter(date_format)

    # ----------------------------
    # Loop over parameters
    # ----------------------------
    for ax, param in zip(axs, params):


        param_color = (
            param_colors.get(param, "tab:purple")
                if param_colors else "tab:purple"
            )

        ax_r = ax.twinx()

        # No loop on external parameter 
        h_param = ax.plot(
            data[time_col],
            data[param],
            linestyle="None",
            marker=marker,
            color=param_color,
            alpha=alpha,
            label=param,          
            )

        # Right axis: CHI2
        # loop on filters
        for f in data[filter_col].unique():
            sub = data[data[filter_col] == f]
            h_chi2 = ax_r.plot(
            sub[time_col],
            sub[chi2_col],
            ".",
            alpha=0.25,
            color=get_filter_color(f)
            )
            
        ax_r.set_yscale("log")

        handles = h_param 
        labels = [h.get_label() for h in handles]

        ax.legend(
                handles,
                labels,
                loc="upper right",
                frameon=True,
        )

        
        # ----------------------------
        # Axis formatting
        # ----------------------------
        ax.set_ylabel(param)

        if param_ylim and param in param_ylim:
            ax.set_ylim(*param_ylim[param])

        if chi2_log:
            ax_r.set_yscale("log")

        if chi2_cut is not None:
            ax_r.axhline(chi2_cut, ls="-.", c="k", alpha=0.5)

        ax_r.set_ylabel(chi2_col)

        title = panel_titles.get(param, param) if panel_titles else param
        ax.set_title(title)

        ax.grid(True, alpha=0.3)

        # Legend only once per panel (filters)
        ax.legend(
            title=filter_col,
            ncol=len(data[filter_col].unique()),
            loc="upper left"
        )

        ax.set_xlabel("time")
        ax.xaxis.set_major_formatter(date_form)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # ----------------------------
    # Shared x-axis
    # ----------------------------
    #axs[-1].set_xlabel("time")
    #axs[-1].xaxis.set_major_formatter(date_form)
    #plt.setp(axs[-1].get_xticklabels(), rotation=45, ha="right")

    # ----------------------------
    # Global title
    # ----------------------------
    if suptitle:
        fig.suptitle(suptitle)

    #if axs is None:
    #    fig.tight_layout()

    return fig, axs



#----------------------------
# param vs chi2
#----------------------------

def plot_param_chi2_correlation_grid(
    df,
    params,
    chi2_col,
    filter_col,
    filter_order=None,        # list of filters in desired column order
    param_ranges=None,        # dict: {param: (xmin, xmax)}
    chi2_range=None,          # (ymin, ymax)
    marker=".",
    alpha=0.3,
    figsize=(4, 3),
):
    """
    Plot correlation between parameters and CHI2_FIT.
    
    Rows: parameters
    Columns: filters
    """

    import matplotlib.pyplot as plt

    # ----------------------------
    # Filter ordering
    # ----------------------------
    if filter_order is None:
        filters = list(df[filter_col].unique())
    else:
        filters = filter_order

    n_params = len(params)
    n_filters = len(filters)

    fig, axs = plt.subplots(
        n_params,
        n_filters,
        figsize=(figsize[0] * n_filters, figsize[1] * n_params),
        sharex=False,
        sharey=False,
        squeeze=False,
    )

    # ----------------------------
    # Loop
    # ----------------------------
    for i, param in enumerate(params):
        for j, f in enumerate(filters):

            ax = axs[i, j]

            sub = df[df[filter_col] == f]

            ax.plot(
                sub[param],
                sub[chi2_col],
                linestyle="None",
                marker=marker,
                alpha=alpha,
                color=get_filter_color(f),
            )

            # ----------------------------
            # Axis limits
            # ----------------------------
            if param_ranges and param in param_ranges:
                ax.set_xlim(*param_ranges[param])

            if chi2_range is not None:
                ax.set_ylim(*chi2_range)

            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")

            # ----------------------------
            # Labels
            # ----------------------------
            ax.set_xlabel(param)
            #if i == n_params - 1:
            #    ax.set_xlabel(param)
            #else:
            #    ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(chi2_col)
            else:
                ax.set_yticklabels([])

           
            # ----------------------------
            # Column titles
            # ----------------------------
            if i == 0:
                ax.set_title(str(f))

    fig.tight_layout()
    return fig, axs


