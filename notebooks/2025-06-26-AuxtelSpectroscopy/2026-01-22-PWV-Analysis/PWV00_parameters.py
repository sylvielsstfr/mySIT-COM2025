# File to centralize config parameters, path and selection files and selection functions
# author : Sylvie Dagoret-Campagne
# affiliation : IJCLab/IN2P3/CNRS
# Creation date : 2026-01-22
# last update : 2026-01-22 : run2026_v01
# last update : 2026-02-07 : add FGCM file
# last update : 2026-02-19 : Corentin files file

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
#version_run = "run_v10" # before v12
#version_run = "run2026_v01" # data 2025 from Jeremy
version_run = "run2026_v02a_cr" # data 2025 from Corentin
#version_run = "run_v12"
# Configuration for the butler repo associated to the version_run

map_run_butler_embargo = { 
                            "run_v10": False,
                            "run_v11": False,
                            "run_v12": False,
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
legendtag = { "run_v10" : "v3_2_0_repo_main_w_2025_42_gains",
              "run_v11" : "v3_2_0_repo_main_w_2025_42_ptc",
              "run_v12" : "spectractorv321testccdgains_all_main_2025data_ptc",
              "run2026_v01" : "v3.2.1 (/repo/main, w_2026_01,ptc)", 
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
    "run_v10" : "u/dagoret/auxtel_run_20251022_w_2025_42_spectractorv32_all_main_data_gains_holoallfilt_a",
    "run_v11" : "u/dagoret/auxtel_run_20251023_w_2025_42_spectractorv32_all_main_2025data_ptc_holoallfilt_a",
    "run_v12" : "u/dagoret/auxtel_run_20251210_w_2025_49_spectractorv321testccdgains_all_main_2025data_ptc_holoallfilt_a",
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
    "run_v10": "../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_u_dagoret_auxtel_run_20251022_w_2025_42_spectractorv32_all_main_data_gains_holoallfilt_a_v1.npy",
    # run v11 : Oct 23 2025
    "run_v11" : "../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_u_dagoret_auxtel_run_20251023_w_2025_42_spectractorv32_all_main_2025data_ptc_holoallfilt_a_v1.npy",
    # run v12 : Dec 12 2025
    "run_v12" : "../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_u_dagoret_auxtel_run_20251210_w_2025_49_spectractorv321testccdgains_all_main_2025data_ptc_holoallfilt_a_v1.npy",
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
    "run_v10":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_run_u_dagoret_auxtel_run_20251022_w_2025_42_spectractorv32_all_main_data_gains_holoallfilt_a_v1_merged.npy",
    "run_v11" : "../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_run_u_dagoret_auxtel_run_20251023_w_2025_42_spectractorv32_all_main_2025data_ptc_holoallfilt_a_v1_merged.npy",
    "run_v12" : "../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_run_u_dagoret_auxtel_run_20251210_w_2025_49_spectractorv321testccdgains_all_main_2025data_ptc_holoallfilt_a_v1_merged.npy",
    "run2026_v01":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_atmosphere_202311_v3.2.1_fixA2fixA1_RobustFit_newThroughputs_merged.parquet.gz",
    #"run2026_v01": "../../../../../fromjeremy/holo_202601/auxtel_atmosphere_202311_v3.2.1_fixA2fixA1_RobustFit_newThroughputs.parquet.gz",
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





def getSelectionCutforPWV(df,
                          pwv_ram_col,
                          pwv_rum_col,
                          chi2_col = "CHI2_FIT",
                          d2ccd_col = "D2CCD",
                          exptime_col = 'EXPTIME',
                          chi2max=CHI2CUT, 
                          pwvmin=PWVMINCUT, 
                          pwvmax = PWVMAXCUT):
    cut =  (df[chi2_col] < chi2max ) & \
    ( df[pwv_ram_col] > pwvmin ) & ( df[pwv_ram_col] < pwvmax ) & \
    ( df[pwv_rum_col] > pwvmin ) & ( df[pwv_rum_col] < pwvmax ) & \
    ( df[d2ccd_col] > DCCDMINCUT ) & ( df[d2ccd_col] < DCCDMAXCUT ) & \
    ( df[exptime_col] > EXPTIMECUT ) 
    return df[cut] 


# Take into account Photometric Repeatability
FLAG_CORRECTFOR_PWV_REPEAT = True
FLAG_CORRECTFOR_PWV_REPEAT_RATIO = False

#SIGMA_PWV_REPEAT = 0.25
SIGMA_PWV_REPEAT = 0.10
FACTORERR_PWV_REPEAT = 10.

# FGCM file
fimename_fgcm = "../../2025-12-04-FGCM/data_FGCM04_ReadMergeAtmParams/fgcm_rdp2_prep_cLSSTCam_runs_DRP_20250417-20251214_DM-53545andDM-53697.csv"

# Merra2 pathname

filename_m2 = "../2025-09-16-SpectroMerra2/MerradataMerged/Merge_inst1_2d_asm_Nx_M2I1NXASM-2021-2025.csv"

# Observation site
site_lsst = EarthLocation.of_site("Cerro Pachon")


