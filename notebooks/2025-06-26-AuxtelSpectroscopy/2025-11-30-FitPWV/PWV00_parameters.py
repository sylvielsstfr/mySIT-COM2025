# File to centralize config parameters, path and selection files and selection functions
# author : Sylvie Dagoret-Campagne
# affiliation : IJCLab/IN2P3/CNRS
# Creation date : 2025-09-20
# last update : 2025-09-30 : run_v7
# last update : 2025-10-25 : run_v11
# input file configuration

from astropy.time import Time
from datetime import datetime
from astropy.coordinates import EarthLocation, AltAz, get_sun
import numpy as np
import astropy.units as u



# Select run version tag to be used in EXTR_viewSpectractor notebooks 
#version_run = "run_v7"
version_run = "run_v10"
# Configuration for the butler repo associated to the version_run

map_run_butler_embargo = { 
                            "run_v1": False,
                            "run_v2": False,
                            "run_v3": True,
                            "run_v4": True,
                            "run_v5": True,
                            "run_v6": False,
                            "run_v7": True,
                            "run_v8": False,
                            "run_v9": False, 
                            "run_v10": False,
                            "run_v11": False,
                         }

FLAG_REPO_EMBARGO = map_run_butler_embargo[version_run]

#('empty~holo4_003','BG40~holo4_003','FELH0600~holo4_003','SDSSr~holo4_003','BG40_65mm_1~holo4_003','OG550_65mm_1~holo4_003') 
#('empty~holo4_003','BG40~holo4_003','FELH0600~holo4_003','SDSSr~holo4_003','BG40_65mm_1~holo4_003','OG550_65mm_1~holo4_003')
FLAG_PWVFILTERS = True
#OZ_FILTER_LIST = ["BG40_65mm_1",]
PWV_FILTER_LIST = ["empty","OG550_65mm_1","SDSSr","FELH0600"]
PWV_FILTEROG550_LIST = ["OG550_65mm_1"]

# Associate the tag to the Spectractor runparameters (to be used in plots)
legendtag = {"run_v1" : "v3.1.0 (/repo/main, w_2025_25,empty,gain)","run_v2":"v3.1.0 (/repo/main, w_2025_25,all-filts,gain)","run_v3":"v3.2.0 (/repo/embargo, w_2025_36,gain),", "run_v4":"v3.2.0 (/repo/embargo,w_2025_36,gain)","run_v5":"v3.2.0  (/repo/embargo,w_2025_36,ptc)","run_v6":"v3.2.0  (/repo/main,w_2025_38,gain)","run_v7":"v3_2_0_repo_main_w_2025_38_gain--v3_2_0_repo_embargo_w_2025_36_ptc","run_v8": "v3_2_0_repo_main_w_2025_42_ptc", "run_v9" : "v3_2_0_repo_main_w_2025_42_gains","run_v10" : "v3_2_0_repo_main_w_2025_42_gains","run_v11" : "v3_2_0_repo_main_w_2025_42_ptc"}

# List of user collection in butler  where the results of spectractor run are
butlerusercollectiondict = {
    # /repo/main
    "run_v1":"u/dagoret/auxtel_run_20250912a",
    # /repo/main
    "run_v2":"u/dagoret/auxtel_run_20250917_w_2025_25_spectractorv31_holoallfilt_a",
    # /repo/embargo
    "run_v3":"u/dagoret/auxtel_run_20250920_w_2025_36_spectractorv32_embargo_gains_holoallfilt_a",
    # /repo/embargo, gains
    "run_v4":"u/dagoret/auxtel_run_20250920_w_2025_36_spectractorv32_embargo_gains_holoallfilt_b",
    # /repo/embargo, ptc 
    "run_v5":"u/dagoret/auxtel_run_20250921_w_2025_36_spectractorv32_embargo_ptc_holoallfilt_a",
    # /repo/main, gains
    "run_v6":"u/dagoret/auxtel_run_20250921_w_2025_38_spectractorv32_main_gains_holoallfilt_a",
     # run_v6 + run_v5 : /repo/main, gains and /repo/embargo
    "run_v7": "u/dagoret/auxtel_run_20250921_w_2025_36_spectractorv32_embargo_ptc_u/dagoret/auxtel_run_20250921_w_2025_38_spectractorv32_main_gains",
        # run_8 : oct 2025 : reprocess all data from 2025 which are in /repo/main now
    "run_v8": "u/dagoret/auxtel_run_20251018_w_2025_42_spectractorv32_main_data2025_ptc_holoallfilt_a",
    # run_v9 : oct 2025 : reprocess all data from 2022-2025 which are in /repo/main now but missing BG40 in 2025
    "run_v9" : "u/dagoret/auxtel_run_20251018_w_2025_42_spectractorv32_all_main_data_gains_holoallfilt_b",
    # run_v10 : oct 2025 : reprocess all data from 2022-2025 which are in /repo/main now with  BG40 in 2025
    "run_v10" : "u/dagoret/auxtel_run_20251022_w_2025_42_spectractorv32_all_main_data_gains_holoallfilt_a",
    "run_v11" : "u/dagoret/auxtel_run_20251023_w_2025_42_spectractorv32_all_main_2025data_ptc_holoallfilt_a",
}


# path of output files Spectractor parameters Extracted from Butler 
extractedfilesdict = {
    # /repo/main
    "run_v1": "../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_atmosphere_20250912a_repomain_v1.npy",
    "run_v2": "../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_20250917_w_2025_25_spectractorv31_holoallfilt_a_repomain_v1.npy",
    # /repo/embargo, gain
    "run_v4":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_u_dagoret_auxtel_run_20250920_w_2025_36_spectractorv32_embargo_gains_holoallfilt_b_v1.npy",
    "run_v5":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_u_dagoret_auxtel_run_20250921_w_2025_36_spectractorv32_embargo_ptc_holoallfilt_a_v1.npy",
    # /repo/main
    "run_v6":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_u_dagoret_auxtel_run_20250921_w_2025_38_spectractorv32_main_gains_holoallfilt_a_v1.npy",
    # run_v6 + run_v5 : /repo/main, gains and /repo/embargo
    "run_v7":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro_joined/auxtel_run_v3_2_0_repo_main_w_2025_38_gain-join-v3_2_0_repo_embargo_w_2025_36_ptc.npy",
    # run_8 : oct 2025 : reprocess all data from 2025 which are in /repo/main now
    "run_v8": "../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_u_dagoret_auxtel_run_20251018_w_2025_42_spectractorv32_main_data2025_ptc_holoallfilt_a_v1.npy",
    # run_v9 : oct 2025 : reprocess all data from 2022-2025 which are in /repo/main now but BG40 missing
    "run_v9": "../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_u_dagoret_auxtel_run_20251018_w_2025_42_spectractorv32_all_main_data_gains_holoallfilt_b_v1.npy",
    # run_v10 : oct 2025 : reprocess all data from 2022-2025 which are in /repo/main now
    "run_v10": "../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_u_dagoret_auxtel_run_20251022_w_2025_42_spectractorv32_all_main_data_gains_holoallfilt_a_v1.npy",
    # run v11 : Oct 23 2025
    "run_v11" : "../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_u_dagoret_auxtel_run_20251023_w_2025_42_spectractorv32_all_main_2025data_ptc_holoallfilt_a_v1.npy",
}


# path of output files Spectractor parameters Extracted from Butler and merged with exposure list
mergedextractedfilesdict = {
    # /repo/main
    "run_v2": "../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_run_20250917_w_2025_25_spectractorv31_holoallfilt_a_repomain_v1_merged.npy",
    # /repo/embargo, gain
    "run_v4":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_run_u_dagoret_auxtel_run_20250920_w_2025_36_spectractorv32_embargo_gains_holoallfilt_b_v1_merged.npy",
    "run_v5":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_run_u_dagoret_auxtel_run_20250921_w_2025_36_spectractorv32_embargo_ptc_holoallfilt_a_v1_merged.npy",
    # /repo/main
    "run_v6":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_run_u_dagoret_auxtel_run_20250921_w_2025_38_spectractorv32_main_gains_holoallfilt_a_v1_merged.npy",
    # run_v6 + run_v5 : /repo/main, gains and /repo/embargo
    "run_v7":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro_joined/auxtel_run_v3_2_0_repo_main_w_2025_38_gain-join-v3_2_0_repo_embargo_w_2025_36_ptc.npy",
    # run_8 : oct 2025 : reprocess all data from 2025 which are in /repo/main now
    "run_v8": "../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_run_u_dagoret_auxtel_run_20251018_w_2025_42_spectractorv32_main_data2025_ptc_holoallfilt_a_v1_merged.npy",
    # run_v9 : oct 2025 : reprocess all data from 2022-2025 which are in /repo/main now BG40 missing in 2025
    "run_v9":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_run_u_dagoret_auxtel_run_20251018_w_2025_42_spectractorv32_all_main_data_gains_holoallfilt_b_v1_merged.npy",
    # run_v10 : oct 2025 : reprocess all data from 2022-2025 which are in /repo/main now
    "run_v10":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_run_u_dagoret_auxtel_run_20251022_w_2025_42_spectractorv32_all_main_data_gains_holoallfilt_a_v1_merged.npy",
    "run_v11" : "../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_run_u_dagoret_auxtel_run_20251023_w_2025_42_spectractorv32_all_main_2025data_ptc_holoallfilt_a_v1_merged.npy",
}


atmfilename = mergedextractedfilesdict[version_run]
tag = legendtag[version_run] 


def DumpConfig():
    """
    Dump the configuration chosen here
    """
    print("**************************************************************************")
    print("*                         DumpConfig()                                   *")
    print(f" \t - version_run = {version_run}")
    print(f" \t - annotation tag = {tag}")
    print(f" \t - extracted file from butler collection = {atmfilename}")
    print("**************************************************************************")
    


# Selection parameters for cuts
DCCDMINFIG = 185.0;
DCCDMAXFIG = 190.0;
DCCDMINCUT = 186.7;
DCCDMAXCUT = 188.0;
#CHI2CUT = 350.;
CHI2CUT = 50.;
EXPTIMECUT = 20.0;    
PWVMINCUT = 0.;
PWVMAXCUT = 20.;
OZMINCUT = 0.;
OZMAXCUT = 650.0;    

def getSelectionCutOld(df_spec, chi2max=20., pwvmin=0.1, pwvmax = 14.9):
    cut =  (df_spec["CHI2_FIT"]<chi2max) & (df_spec["PWV [mm]_x"] > pwvmin) & (df_spec["PWV [mm]_x"] < pwvmax) 
     #(df_spec["ozone [db]_y"] > ozmin) & (df_spec["ozone [db]_y"] < ozmax) & (df_spec["TARGET"] == "HD185975")
    return cut

    
def getSelectionCut(df_spec, chi2max=CHI2CUT, pwvmin=PWVMINCUT, pwvmax = PWVMAXCUT,ozmin=OZMINCUT,ozmax=OZMAXCUT):
    cut =  (df_spec["CHI2_FIT"]<chi2max) & (df_spec["PWV [mm]_x"] > pwvmin) & (df_spec["PWV [mm]_x"] < pwvmax) & (df_spec["D2CCD"]>DCCDMINCUT) &  (df_spec["D2CCD"]<DCCDMAXCUT) & \
    (df_spec['EXPTIME'] > EXPTIMECUT ) 
    return cut 

def getSelectionCutNoPolar(df_spec, chi2max=CHI2CUT, pwvmin=PWVMINCUT, pwvmax=PWVMAXCUT,ozmin=OZMINCUT,ozmax=OZMAXCUT):
    cut =  (df_spec["CHI2_FIT"]<chi2max) & (df_spec["PWV [mm]_x"] > pwvmin) & (df_spec["PWV [mm]_x"] < pwvmax) & (df_spec["D2CCD"]>DCCDMINCUT) &  (df_spec["D2CCD"]<DCCDMAXCUT) & \
    (df_spec['EXPTIME'] > EXPTIMECUT ) &   (df_spec["TARGET"] != "HD185975")
    return cut

def getSelectionCutWithPolar(df_spec, chi2max=CHI2CUT, pwvmin=PWVMINCUT, pwvmax=PWVMAXCUT,ozmin=OZMINCUT,ozmax=OZMAXCUT):
    cut =  (df_spec["CHI2_FIT"]<chi2max) & (df_spec["PWV [mm]_x"] > pwvmin) & (df_spec["PWV [mm]_x"] < pwvmax) & (df_spec["D2CCD"]>DCCDMINCUT) &  (df_spec["D2CCD"]<DCCDMAXCUT) & \
    (df_spec['EXPTIME'] > EXPTIMECUT ) &   (df_spec["TARGET"] == "HD185975")
    return cut


# Take into account Photometric Repeatability
FLAG_CORRECTFOR_PWV_REPEAT = True
FLAG_CORRECTFOR_PWV_REPEAT_RATIO = False

#SIGMA_PWV_REPEAT = 0.25
SIGMA_PWV_REPEAT = 0.10
FACTORERR_PWV_REPEAT = 10.



# Merra2 pathname

filename_m2 = "../2025-09-16-SpectroMerra2/MerradataMerged/Merge_inst1_2d_asm_Nx_M2I1NXASM-2021-2025.csv"

# Observation site
site_lsst = EarthLocation.of_site("Cerro Pachon")


def get_astronomical_midnight(location: EarthLocation, date, n_grid=1000):
    """
    Transit inférieur du Soleil (min altitude) pour date & site avec astropy pur.
    
    Parameters
    ----------
    location : EarthLocation
        Site d’observation (lat, lon, hauteur)
    date : str or Time
        Date de référence (ex: "2025-09-23")
    n_grid : int
        Nombre de points sur 24h à évaluer pour estimer le minimum
    
    Returns
    -------
    Time
        Temps UTC quand le Soleil a l’altitude minimale
    """


    if isinstance(date, datetime):
        t = Time(date, scale="utc")
    elif hasattr(date, "strftime"):  # ex: datetime.date
        t = Time(date.strftime("%Y-%m-%d"), scale="utc")
    elif not isinstance(date, Time):
        t = Time(date, scale="utc")
    else:
        t = date
    
    date = t + 1 * u.day
    
    # Définir intervalle de ~24h autour de la date
    t0 = date - 12  * u.hour
    t1 = date + 12  * u.hour
    
    # Grille de temps
    times = Time(np.linspace(t0.jd, t1.jd, n_grid), format='jd', scale='utc')
    
    # Position du Soleil
    sun = get_sun(times)
    
    # AltAz
    aa = AltAz(obstime=times, location=location)
    sun_altaz = sun.transform_to(aa)
    
    # Trouver index du min
    idx_min = np.argmin(sun_altaz.alt)
    t_min = times[idx_min]
    return t_min.to_datetime()
