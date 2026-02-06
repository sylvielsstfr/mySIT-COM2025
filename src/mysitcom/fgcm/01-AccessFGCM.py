#!/usr/bin/env python

# ## Access to FGCM data
#
# - creation date : 2025-12-04
# -  last update : 2025-12-05
# -  last update : 2025-12-08 : add more FGCM products
# -  last update : 2025-12-12 : collection = "LSSTCam/runs/DRP/20250417_20250921/w_2025_49/DM-53545"
# - last update : 2026-02-06 : transform notebook in script to run interactively
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


import textwrap

import lsst.geom as geom
import numpy as np
from lsst.daf.butler import Butler

REPO_URI  = "dp2_prep"
#collection = "LSSTCam/runs/DRP/20250417_20250723/d_2025_11_21/DM-53374" # 2025-12-05
#collection = "LSSTCam/runs/DRP/20250417_20250921/w_2025_49/DM-53545" # 2025-12-12
#collection = "LSSTCam/runs/DRP/20250417_20250921/w_2025_49/DM-53545" # 2026-02-06 : collection utilisée pour l'extraction interactive des données FGCM
#collection = "LSSTCam/runs/DRP/20250515-20251214/v30_0_0_rc2/DM-53697" # 2026-02-06 : collection utilisée pour l'extraction interactive des données FGCM (DRP v30.0.0-rc2)
collection = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/"
butler = Butler(REPO_URI, collections=collection)
registry = butler.registry
dataset_types = list(registry.queryDatasetTypes())


# --- NOUVEAU FILTRE : Types de données pertinents ---
# Classes de stockage que nous voulons conserver (tables de données)
ALLOWED_STORAGE_CLASSES = ['DataFrame', 'SourceCatalog', 'Catalog','ArrowAstropy']
#ALLOWED_STORAGE_CLASSES = ['DataFrame']

# Mots-clés pour filtrer les datasets liés à la détection et à l'alerte
ALLOWED_KEYWORDS = ['fgcm', 'FGCM','transmission']



keep_dataset_types = []
keep_dataset_dimensions = {}

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

    #if is_relevant_keyword and is_relevant_storage:
    if is_relevant_keyword:
        # Utilisation de textwrap pour gérer les noms de datasets longs sans tronquer la sortie
        wrapped_name = textwrap.shorten(ds_name, width=40, placeholder='...')
        print(f"  - **{wrapped_name:40s}** : stored like a {storage_class_name}")
        keep_dataset_types.append(ds_name)
        required_dimensions = list(ds_type.dimensions.names)
        keep_dataset_dimensions[ds_name]= required_dimensions

print("\n--- Liste des catalogues à utiliser pour l'extraction de données : ---")
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



print(keep_dataset_dimensions)


# ## Access to transmissions
#
# ```
# 'transmission_atmosphere_fgcm': ['band',
#   'instrument',
#   'day_obs',
#   'physical_filter',
#   'visit'],
# ```


datasetType = "transmission_atmosphere_fgcm"
refs = butler.registry.queryDatasets(datasetType)



refs = list(refs)
the_ref = refs[0]
print(the_ref.dataId)
print(the_ref.datasetType.dimensions)


transm = butler.get(the_ref)


dir(transm)

print(transm.getWavelengthBounds())
print(transm.getThroughputAtBounds())

# wavelengths en nm :
w = np.linspace(300., 11000., 50)

# position arbitraire (par exemple centre du détecteur)
pos = geom.Point2D(2000, 2000)

T = transm.sampleAt(pos, w)
