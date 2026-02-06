#!/usr/bin/env python


# ## Extract FGCM data
#
# - creation date : 2025-12-05
# - last update : 2025-12-08 : add more FGCM products
# - last update : 2025-12-12 : collection = "LSSTCam/runs/DRP/20250417_20250921/w_2025_49/DM-53545"
# - last update : 2026-02-06 : transform notebook in script to run interactively
# - See list of DRP runs : https://rubinobs.atlassian.net/wiki/spaces/DM/pages/661192727/LSSTCam+Intermittent+DRP+Runs

# 1. Où sont stockées les mesures FGCM dans Rubin ?
#
# Dans Rubin/LSST, les produits de FGCM sont générés pendant le photometric calibration step du pipeline DRP. Ils vivent dans le Butler sous forme de datasets, typiquement appelés :
#
# - fgcmParameters
# - fgcmLookUpTable
# - fgcmVisitCatalog
# - fgcmStarObservations
# - fgcmAtmosphereParameters
# - fgcmZeroPoints
#
# (et parfois) calib_photometry / photometricCatalog
#
# Ils sont stockés dans une collection, par exemple :
# - LSSTCam/runs/DRP/xxxxxx/DM-xxxxx
# - dp02/runs/…
# - dp2_prep/LSSTCam/... (comme dans ton URL)

# ### Links
# - doc lsst-pipelines : https://pipelines.lsst.io/v/v23_0_0/modules/lsst.fgcmcal/
# - github : https://github.com/lsst/fgcmcal
# - runs : https://rubinobs.atlassian.net/wiki/spaces/DM/pages/661192727/LSSTCam+Intermittent+DRP+Runs
# - plot-Navigator: https://usdf-rsp.slac.stanford.edu/plot-navigator/collection/dp2_prep/LSSTCam%2Fruns%2FDRP%2F20250417_20250723%2Fd_2025_11_21%2FDM-53374

# ## 📄 Principaux dataset_types associés à FGCM (à titre d’exemple)
#
# Voici une liste non exhaustive de dataset_types liés à FGCM, tels qu’utilisés/produits par le pipeline :
#
# dataset_type	Description / rôle
# - `fgcmLookUpTable` :	Table de correspondance (look-up table) combinant les effets instrumentaux (transmissions, filtres) et atmosphériques — produite par la tâche de type “Make LUT”.
# - `fgcmStandardStars`: 	Catalog de “référence d’étoiles standard” utilisés dans le fit FGCM (input pour la calibration absolue).
# - `fgcmZeropoints`:	Catalogue des “zero-points” photométriques calculés par le fit FGCM — un des principaux outputs.
# - `fgcmAtmosphereParameters`: 	Catalogue des paramètres d’atmosphère (extinction, transmission, conditions …) associés aux visites / expositions.
# - `fgcm_stars`:	Catalogue de référence d’étoiles calibrées FGCM, utilisable pour l’étalonnage photométrique absolu.
# - `fgcm_photoCalib`: 	Produits de calibration photométrique (PhotoCalib) générés pour être utilisés dans le pipeline LSST downstream (coadd, mesures, etc.).
#



import os
import textwrap
from datetime import datetime

import astropy.units as u

# Remove to run faster the notebook
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.table import join
from astropy.time import Time
from lsst.daf.butler import Butler

# ## Configuration

plt.rcParams["figure.figsize"] = (16,8)
plt.rcParams["axes.labelsize"] = 'xx-large'
plt.rcParams['axes.titlesize'] = 'xx-large'
plt.rcParams['xtick.labelsize']= 'xx-large'
plt.rcParams['ytick.labelsize']= 'xx-large'
plt.rcParams["legend.fontsize"] = "xx-large"


# In[ ]:


# Rubin-LSST / Cerro Pachón
lsst = EarthLocation(lat=-30.2417*u.deg, lon=-70.7366*u.deg, height=2663*u.m)



# where are stored the figures
pathfigs = "figs_FGCM02_ExtractAtmParams"
prefix = "fgcm02"
if not os.path.exists(pathfigs):
    os.makedirs(pathfigs)
figtype = ".png"



REPO_URI  = "dp2_prep"
#collection = "LSSTCam/runs/DRP/20250417_20250723/d_2025_11_21/DM-53374" # 2025-12-05
# used in december 2025
#collection = "LSSTCam/runs/DRP/20250417_20250921/w_2025_49/DM-53545" # 2025-12-12
# used in february 2026
#collection = "LSSTCam/runs/DRP/20250515-20251214/v30_0_0_rc2/DM-53697"
collection = "dp2_prep: LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/" # 2026-02-06 : collection utilisée pour l'extraction interactive des données FGCM (DRP v30.0.0-rc2)

strcollection = collection.replace("/","_")
strrepo = REPO_URI.replace("/","_")
suptitle = f"repo {REPO_URI}, coll = {collection}"
butler = Butler(REPO_URI, collections=collection)
registry = butler.registry
dataset_types = list(registry.queryDatasetTypes())



# --- NOUVEAU FILTRE : Types de données pertinents ---
# Classes de stockage que nous voulons conserver (tables de données)
ALLOWED_STORAGE_CLASSES = ['DataFrame', 'SourceCatalog', 'Catalog','ArrowAstropy','TransmissionCurve']
#ALLOWED_STORAGE_CLASSES = ['DataFrame']

# Mots-clés pour filtrer les datasets liés à la détection et à l'alerte
ALLOWED_KEYWORDS = ['fgcm', 'FGCM']


def solar_midnight_utc(day_mjd, location):
    """
    Retourne l'heure UTC approximative du min du Soleil (culmination la plus basse)
    pour le site donné et un jour MJD.
    """
    # Création d'une grille de temps toutes les 5 minutes sur ce jour
    t_start = Time(day_mjd, format='mjd')
    t_grid = t_start + np.arange(0, 1, 5/1440)  # 1 jour = 1440 min

    # Calcul altitude du Soleil
    altaz = AltAz(obstime=t_grid, location=location)
    sun_alt = get_sun(t_grid).transform_to(altaz).alt

    # Trouver l'heure du min
    idx_min = np.argmin(sun_alt)
    return t_grid[idx_min]  # Retourne un Time object



# Palette par filtre
default_filter_colors = {
    "u_24": "tab:blue",
    "g_6":  "tab:green",
    "r_57": "tab:red",
    "i_39": "tab:orange",
    "z_20": "tab:gray",
    "y_10": "black"
}

def plot_atm_parameter(t_join, param="pwv", filter_colors=None):
    """
    Trace un paramètre atmosphérique par date, par filtre,
    avec bandes grises = nuit astronomique au site Rubin-LSST.

    Paramètres
    ----------
    t_join : astropy.Table
        Table jointe avec colonnes 'mjd', 'physicalFilter' et le paramètre choisi
    param : str
        Nom du paramètre à tracer ('pwv', 'o3', 'tau', etc.)
    filter_colors : dict, optional
        Dictionnaire {filter_name: couleur}, sinon palette par défaut
    """
    if filter_colors is None:
        filter_colors = default_filter_colors

    mjd = t_join['mjd']
    filters = t_join['physicalFilter']
    values = t_join[param]

    mask_valid = np.isfinite(values)
    dates_utc = Time(mjd, format='mjd').to_datetime()

    filter_order = list(filter_colors.keys())

    plt.figure(figsize=(18,8))

    # Scatter par filtre
    for f in filter_order:
        m = (filters == f) & mask_valid
        if np.sum(m) > 0:
            plt.scatter(dates_utc[m], values[m], s=12, alpha=0.6,
                        color=filter_colors[f], label=f)

    # Fonctions auxiliaires
    def night_astronomical_utc(day_mjd, location):
        t_start = Time(day_mjd, format='mjd')
        t_grid = t_start + np.arange(0, 1.5, 5/1440)  # 1.5 jour pour capturer la nuit complète
        altaz = AltAz(obstime=t_grid, location=location)
        sun_alt = get_sun(t_grid).transform_to(altaz).alt
        mask_night = sun_alt < -18*u.deg
        night_times = t_grid[mask_night]
        if len(night_times) == 0:
            return None, None
        return night_times[0], night_times[-1]

    # Bandes grises = nuit astronomique
    start_day = int(np.floor(mjd.min()))
    end_day   = int(np.ceil(mjd.max()))
    all_days = np.arange(start_day, end_day, 1)

    for day_mjd in all_days:
        start_night, end_night = night_astronomical_utc(day_mjd, lsst)
        if start_night is not None:
            plt.axvspan(start_night.datetime, end_night.datetime,
                        color='gray', alpha=0.05)

    # Format axe X
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d:%H'))
    # Locator pour avoir au moins 1 tick par semaine
    #ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
    #ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=7, maxticks=15))
    # Tick mineur : 1 par jour
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))

    plt.xticks(rotation=45)

    plt.xlabel("Date (UTC)")
    plt.ylabel(f"{param.upper()}")
    plt.title(f"{param.upper()} vs Date (colored by filter)\nGray = astronomical night LSST")
    plt.legend(title="Filter", markerscale=1.5)
    plt.grid(True, alpha=0.3)
    plt.suptitle(suptitle)
    plt.tight_layout()
    figname =f"{pathfigs}/{prefix}_{param}"+figtype
    plt.savefig(figname)
    plt.show()


# ## Start


keep_dataset_types = []

# loop on datasettypes
for ds_type in dataset_types:

    ds_name = ds_type.name

    # Correction de l'erreur: Utilisation d'un bloc try/except pour gérer les
    # "KeyError" qui se produisent lorsque le Butler ne peut pas résoudre la classe de stockage (ex: 'SpectractorWorkspace').
    storage_class_name = "N/A"
    try:
        if ds_type.storageClass:
            storage_class_name = ds_type.storageClass.name
    except KeyError:
        # La classe existe dans le registre mais le module Python n'est pas chargé
        storage_class_name = "UNRESOLVED_CLASS"
    except Exception as e:
        # Pour tout autre type d'erreur
        print(f"Avertissement: Échec de la résolution de la classe de stockage pour {ds_name}. Erreur: {e}")


    # Filtre 1: Doit contenir un mot-clé pertinent dans le nom
    # Utilisation de .upper() pour une comparaison insensible à la casse dans le nom du dataset
    is_relevant_keyword = any(keyword.upper() in ds_name.upper() for keyword in ALLOWED_KEYWORDS)

    # Filtre 2: Doit être une classe de stockage de type catalogue/table
    is_relevant_storage = storage_class_name in ALLOWED_STORAGE_CLASSES

    if is_relevant_keyword and is_relevant_storage:
        # Utilisation de textwrap pour gérer les noms de datasets longs sans tronquer la sortie
        wrapped_name = textwrap.shorten(ds_name, width=40, placeholder='...')
        print(f"  - **{wrapped_name:40s}** : stored like a {storage_class_name}")
        keep_dataset_types.append(ds_name)
        required_dimensions = list(ds_type.dimensions.names)
        #keep_dataset_dimensions.append(required_dimensions)

print("\n---List of data for extraction: ---")
print(keep_dataset_types)


# LUT instrumentale
lut = butler.get("fgcmLookUpTable")
t_lut = lut.asAstropy()
# visit list
visits = butler.get("fgcmVisitCatalog")
t_visit = visits.asAstropy()
# Atmosphere
fitparam = butler.get("fgcm_Cycle5_FitParameters")
t_fitparam = fitparam.asAstropy()
atmparam = butler.get('fgcm_Cycle5_AtmosphereParameters')
t_atmparam = atmparam.asAstropy()
# Zero pointatmparam
zpt = butler.get('fgcm_Cycle5_Zeropoints')
t_zpt = zpt.asAstropy()
# Catalogue des observations d'étoiles utilisées
#stars = butler.get("fgcmStarObservations")



t_join = join(t_visit, t_atmparam, keys="visit", join_type="left")




t_join = join(
    t_visit,
    t_atmparam,
    keys="visit",
    join_type="left",
    table_names=["visit", "atm"],
    uniq_col_name="{col_name}_{table_name}"
)


print(t_join.colnames)
print(t_join[:2])



print([name for name in t_join.colnames if "pmb" in name])


# ### Check the physical filters


unique_filters = np.unique(t_join['physicalFilter'])
print(unique_filters)


# ### Check missing




missing = set(t_visit['visit']) - set(t_atmparam['visit'])
print(f"{len(missing)} visits non présents dans atmparam")
print(list(missing)[:10])


# ## Save in a file



# Choix du format : "fits" ou "ecsv"
format_save = "fits"  # ou "ecsv"

# Date et heure actuelle pour versionner le fichier
now = datetime.utcnow()  # UTC
timestamp = now.strftime("%Y%m%d_%H%M%S")  # ex: 20251205_094512

# Nom de fichier
filename = f"fgcm_r{strrepo}_c{strcollection}_{timestamp}.{format_save}"

# Sauvegarde
if format_save.lower() == "fits":
    t_join.write(filename, overwrite=True)
elif format_save.lower() == "ecsv":
    t_join.write(filename, format="ascii.ecsv", overwrite=True)
else:
    raise ValueError("Format non supporté. Choisir 'fits' ou 'ecsv'.")

print(f"Table sauvegardée dans {filename}")


# suppress plotting  in interactive mode
if 0:
    # ## Plots
    # ### Plot PWV
    plot_atm_parameter(t_join, param="pwv")
    # ### Plot Ozone
    plot_atm_parameter(t_join, param="o3")
    # ## Plots
    # ### Plot tau
    plot_atm_parameter(t_join, param="tau")
    # ### Plot alpha
    plot_atm_parameter(t_join, param="alpha")


exit(0)
