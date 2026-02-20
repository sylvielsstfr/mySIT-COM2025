# File to centralize config parameters, path and selection files and selection functions
# author : Sylvie Dagoret-Campagne
# affiliation : IJCLab/IN2P3/CNRS
# Creation date : 2026-01-12
# last update : 2026-01-12 : run2026_v01
# last update : 2026-02-20 : Corentin files file

# Runs from Corentin Ravoux added 2026-02-19
#"run2026_v02a_cr : 'auxtel_atmosphere_feb26_gaiaspec_calspectarget_calspecthroughput.npy'
#"run2026_v02b_cr :'auxtel_atmosphere_feb26_gaiaspec_calspecgaiatarget_calspecthroughput_preoct23.npy',
#"run2026_v02c_cr : 'auxtel_atmosphere_feb26_gaiaspec_calspecgaiatarget_calspecthroughput.npy',

#"run2026_v02d_cr :'auxtel_atmosphere_feb26_gaiaspec_gaiatarget_calspecthroughput.npy',

#"run2026_v02e_cr :'auxtel_atmosphere_feb26_gaiaspec_gaiatarget_calspecthroughput_m1corr.npy',
#"run2026_v02f_cr : 'auxtel_atmosphere_feb26_gaiaspec_gaiatarget_calspecthroughput_m2corr.npy',
#"run2026_v02g_cr : 'auxtel_atmosphere_feb26_gaiaspec_gaiatarget_calspecthroughput_m3corr.npy',



from astropy.time import Time
from datetime import datetime
from astropy.coordinates import EarthLocation, AltAz, get_sun
import numpy as np
import astropy.units as u



# Select run version tag to be used in EXTR_viewSpectractor notebooks 
#version_run = "run_v7"
#version_run = "run_v11" # before v12
#version_run = "run2026_v01" # data 2025 # from Jeremy
version_run = "run2026_v02a_cr" # data 2025 from Corentin
# Configuration for the butler repo associated to the version_run

map_run_butler_embargo = { 
                            "run2026_v01": False,
                            "run2026_v02a_cr": False,
                            "run2026_v02b_cr": False,
                            "run2026_v02c_cr": False,
                            "run2026_v02d_cr": False,
                            "run2026_v02e_cr": False,
                            "run2026_v02f_cr": False,
                            "run2026_v02g_cr": False,
                         }

FLAG_REPO_EMBARGO = map_run_butler_embargo[version_run]

#('empty~holo4_003','BG40~holo4_003','FELH0600~holo4_003','SDSSr~holo4_003','BG40_65mm_1~holo4_003','OG550_65mm_1~holo4_003') 
#('empty~holo4_003','BG40~holo4_003','FELH0600~holo4_003','SDSSr~holo4_003','BG40_65mm_1~holo4_003','OG550_65mm_1~holo4_003')
FLAG_PWVFILTERS = True
#OZ_FILTER_LIST = ["BG40_65mm_1",]
PWV_FILTER_LIST = ["empty","OG550_65mm_1","SDSSr","FELH0600"]
PWV_FILTEROG550_LIST = ["OG550_65mm_1"]

# Associate the tag to the Spectractor runparameters (to be used in plots)
legendtag = {"run2026_v01" : "v3.2.1 (/repo/main, w_2026_01,ptc)",
              "run2026_v02a_cr" : "cr_feb26_gaiaspec_calspectarget_calspecthroughput",
              "run2026_v02b_cr" : "cr_feb26_gaiaspec_calspecgaiatarget_calspecthroughput_preoct23",
              "run2026_v02c_cr" : "cr_feb26_gaiaspec_calspecgaiatarget_calspecthroughput",
              "run2026_v02d_cr" : "cr_feb26_gaiaspec_gaiatarget_calspecthroughput",
              "run2026_v02e_cr" : "cr_feb26_gaiaspec_gaiatarget_calspecthroughput_m1corr",
              "run2026_v02f_cr" : "cr_feb26_gaiaspec_gaiatarget_calspecthroughput_m2corr",
              "run2026_v02g_cr" : "cr_feb26_gaiaspec_gaiatarget_calspecthroughput_m3corr"
            }

# List of user collection in butler  where the results of spectractor run are
butlerusercollectiondict = {
    # /repo/main
    "run2026_v01":"u/jneveu/auxtel_atmosphere_202311_v3.2.1_fixA2fixA1_RobustFit_newThroughputs/20260102T180833Z",
    "run2026_v02a_cr" : "cr_feb26_gaiaspec_calspectarget_calspecthroughput",
    "run2026_v02b_cr" : "cr_feb26_gaiaspec_calspecgaiatarget_calspecthroughput_preoct23",
    "run2026_v02c_cr" : "cr_feb26_gaiaspec_calspecgaiatarget_calspecthroughput",
    "run2026_v02d_cr" : "cr_feb26_gaiaspec_gaiatarget_calspecthroughput",
    "run2026_v02e_cr" : "cr_feb26_gaiaspec_gaiatarget_calspecthroughput_m1corr",
    "run2026_v02f_cr" : "cr_feb26_gaiaspec_gaiatarget_calspecthroughput_m2corr",
    "run2026_v02g_cr" : "cr_feb26_gaiaspec_gaiatarget_calspecthroughput_m3corr"   
}


# path of output files Spectractor parameters Extracted from Butler 
extractedfilesdict = {
    # /repo/main
    "run2026_v01": "../../../../../fromjeremy/holo_202601/auxtel_atmosphere_202311_v3.2.1_fixA2fixA1_RobustFit_newThroughputs.parquet.gz",
    "run2026_v02a_cr" : "../../../../../fromcorentin/holo_202602/auxtel_atmosphere_feb26_gaiaspec_calspectarget_calspecthroughput.npy",
    "run2026_v02b_cr" : "../../../../../fromcorentin/holo_202602/auxtel_atmosphere_feb26_gaiaspec_calspecgaiatarget_calspecthroughput_preoct23.npy",
    "run2026_v02c_cr" : "../../../../../fromcorentin/holo_202602/auxtel_atmosphere_feb26_gaiaspec_calspecgaiatarget_calspecthroughput.npy",
    "run2026_v02d_cr" : "../../../../../fromcorentin/holo_202602/auxtel_atmosphere_feb26_gaiaspec_gaiatarget_calspecthroughput.npy",
    "run2026_v02e_cr" : "../../../../../fromcorentin/holo_202602/auxtel_atmosphere_feb26_gaiaspec_gaiatarget_calspecthroughput_m1corr.npy",
    "run2026_v02f_cr" : "../../../../../fromcorentin/holo_202602/auxtel_atmosphere_feb26_gaiaspec_gaiatarget_calspecthroughput_m2corr.npy",
    "run2026_v02g_cr" : "../../../../../fromcorentin/holo_202602/auxtel_atmosphere_feb26_gaiaspec_gaiatarget_calspecthroughput_m3corr.npy"
}


# path of output files Spectractor parameters Extracted from Butler and merged with exposure list
mergedextractedfilesdict = {
    # /repo/main
    "run2026_v01":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_atmosphere_202311_v3.2.1_fixA2fixA1_RobustFit_newThroughputs_merged.parquet.gz",
    "run2026_v02a_cr" : "../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_atmosphere_feb26_gaiaspec_calspectarget_calspecthroughput_merged.npy",
    "run2026_v02b_cr" : "../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_atmosphere_feb26_gaiaspec_calspecgaiatarget_calspecthroughput_preoct23_merged.npy",
    "run2026_v02c_cr" : "../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_atmosphere_feb26_gaiaspec_calspecgaiatarget_calspecthroughput_merged.npy",
    "run2026_v02d_cr" : "../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_atmosphere_feb26_gaiaspec_gaiatarget_calspecthroughput_merged.npy",
    "run2026_v02e_cr" : "../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_atmosphere_feb26_gaiaspec_gaiatarget_calspecthroughput_m1corr_merged.npy",
    "run2026_v02f_cr" : "../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_atmosphere_feb26_gaiaspec_gaiatarget_calspecthroughput_m2corr_merged.npy",
    "run2026_v02g_cr" : "../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_atmosphere_feb26_gaiaspec_gaiatarget_calspecthroughput_m3corr_merged.npy"
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
