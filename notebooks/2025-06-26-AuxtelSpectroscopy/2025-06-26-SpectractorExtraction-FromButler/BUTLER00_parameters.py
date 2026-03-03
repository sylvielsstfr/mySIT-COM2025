# input file configuration

# Select run version tag to be used in EXTR_viewSpectractor notebooks 
#version_run = "run2026_v01"
version_run = "run2026_v02a_cr"
# Configuration for the butler repo associated to the version_run

# Runs from Corentin Ravoux added 2026-02-18
#"run2026_v02a_cr : 'auxtel_atmosphere_feb26_gaiaspec_calspectarget_calspecthroughput.npy'
#"run2026_v02b_cr :'auxtel_atmosphere_feb26_gaiaspec_calspecgaiatarget_calspecthroughput_preoct23.npy',
#"run2026_v02c_cr : 'auxtel_atmosphere_feb26_gaiaspec_calspecgaiatarget_calspecthroughput.npy',

#"run2026_v02d_cr :'auxtel_atmosphere_feb26_gaiaspec_gaiatarget_calspecthroughput.npy',

#"run2026_v02e_cr :'auxtel_atmosphere_feb26_gaiaspec_gaiatarget_calspecthroughput_m1corr.npy',
#"run2026_v02f_cr : 'auxtel_atmosphere_feb26_gaiaspec_gaiatarget_calspecthroughput_m2corr.npy',
#"run2026_v02g_cr : 'auxtel_atmosphere_feb26_gaiaspec_gaiatarget_calspecthroughput_m3corr.npy',





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

# Associate the tag to the Spectractor runparameters (to be used in plots)
legendtag = {"run_v1" : "v3.1.0 (/repo/main, w_2025_25,empty,gain)","run_v2":"v3.1.0 (/repo/main, w_2025_25,all-filts,gain)","run_v3":"v3.2.0 (/repo/embargo, w_2025_36,gain),", "run_v4":"v3.2.0 (/repo/embargo,w_2025_36,gain)","run_v5":"v3.2.0  (/repo/embargo,w_2025_36,ptc)","run_v6":"v3.2.0  (/repo/main,w_2025_38,gain)","run_v7":"v3_2_0_repo_main_w_2025_38_gain-v3_2_0_repo_embargo_w_2025_36_ptc",
"run_v8": "v3_2_0_repo_main_w_2025_42_ptc", "run_v9" : "v3_2_0_repo_main_w_2025_42_gains","run_v10" : "v3_2_0_repo_main_w_2025_42_gains",
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
    "run_7": "u/dagoret/auxtel_run_20250921_w_2025_36_spectractorv32_embargo_ptc_u/dagoret/auxtel_run_20250921_w_2025_38_spectractorv32_main_gains",
    # run_8 : oct 2025 : reprocess all data from 2025 which are in /repo/main now
    "run_v8": "u/dagoret/auxtel_run_20251018_w_2025_42_spectractorv32_main_data2025_ptc_holoallfilt_a",
    # run_v9 : oct 2025 : reprocess all data from 2022-2025 which are in /repo/main now but missing BG40 in 2025
    "run_v9" : "u/dagoret/auxtel_run_20251018_w_2025_42_spectractorv32_all_main_data_gains_holoallfilt_b",
    # run_v10 : oct 2025 : reprocess all data from 2022-2025 which are in /repo/main now with  BG40 in 2025
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
    "run_v1": "../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_atmosphere_20250912a_repomain_v1.npy",
    "run_v2": "../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_20250917_w_2025_25_spectractorv31_holoallfilt_a_repomain_v1.npy",
    # /repo/embargo, gain
    "run_v4":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_u_dagoret_auxtel_run_20250920_w_2025_36_spectractorv32_embargo_gains_holoallfilt_b_v1.npy",
    "run_v5":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_u_dagoret_auxtel_run_20250921_w_2025_36_spectractorv32_embargo_ptc_holoallfilt_a_v1.npy",
    # /repo/main
    "run_v6":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_u_dagoret_auxtel_run_20250921_w_2025_38_spectractorv32_main_gains_holoallfilt_a_v1.npy",
     # run_v6 + run_v5 : /repo/main, gains and /repo/embargo
    "run_7":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro_joined/auxtel_run_v3_2_0_repo_main_w_2025_38_gain-join-v3_2_0_repo_embargo_w_2025_36_ptc.npy", 
    # run_8 : oct 2025 : reprocess all data from 2025 which are in /repo/main now
    "run_v8": "../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_u_dagoret_auxtel_run_20251018_w_2025_42_spectractorv32_main_data2025_ptc_holoallfilt_a_v1.npy",
    # run_v9 : oct 2025 : reprocess all data from 2022-2025 which are in /repo/main now but BG40 missing
    "run_v9": "../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_u_dagoret_auxtel_run_20251018_w_2025_42_spectractorv32_all_main_data_gains_holoallfilt_b_v1.npy",
    # run_v10 : oct 2025 : reprocess all data from 2022-2025 which are in /repo/main now
    "run_v10": "../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_u_dagoret_auxtel_run_20251022_w_2025_42_spectractorv32_all_main_data_gains_holoallfilt_a_v1.npy",
    # run v11 : Oct 23 2025
    "run_v11" : "../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_u_dagoret_auxtel_run_20251023_w_2025_42_spectractorv32_all_main_2025data_ptc_holoallfilt_a_v1.npy",
    # run v12 : Dec 12 2025
    "run_v12" : "../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_u_dagoret_auxtel_run_20251210_w_2025_49_spectractorv321testccdgains_all_main_2025data_ptc_holoallfilt_a_v1.npy",
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

# path of output files Spectractor parameters Extracted from Butler and merged with exposure list from butler registry
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
    "run_v12" : "../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_run_u_dagoret_auxtel_run_20251210_w_2025_49_spectractorv321testccdgains_all_main_2025data_ptc_holoallfilt_a_v1_merged.npy",
    "run2026_v01":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_atmosphere_202311_v3.2.1_fixA2fixA1_RobustFit_newThroughputs_merged.parquet.gz",
    "run2026_v02a_cr" : "../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_atmosphere_feb26_gaiaspec_calspectarget_calspecthroughput_merged.npy",
    "run2026_v02b_cr" : "../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_atmosphere_feb26_gaiaspec_calspecgaiatarget_calspecthroughput_preoct23_merged.npy",
    "run2026_v02c_cr" : "../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_atmosphere_feb26_gaiaspec_calspecgaiatarget_calspecthroughput_merged.npy",
    "run2026_v02d_cr" : "../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_atmosphere_feb26_gaiaspec_gaiatarget_calspecthroughput_merged.npy",
    "run2026_v02e_cr" : "../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_atmosphere_feb26_gaiaspec_gaiatarget_calspecthroughput_m1corr_merged.npy",
    "run2026_v02f_cr" : "../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_atmosphere_feb26_gaiaspec_gaiatarget_calspecthroughput_m2corr_merged.npy",
    "run2026_v02g_cr" : "../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_atmosphere_feb26_gaiaspec_gaiatarget_calspecthroughput_m3corr_merged.npy"
    
}

mergedtofindmissings = {
    "run_v4":"../2025-06-26-SpectractorExtraction-FromButler/data/missing_merged/exposurelist_and_auxtel_run_u_dagoret_auxtel_run_20250920_w_2025_36_spectractorv32_embargo_gains_holoallfilt_b_v1.npy",
    "run_v5":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_u_dagoret_auxtel_run_20250921_w_2025_36_spectractorv32_embargo_ptc_holoallfilt_a_v1.npy",
    "run_v5":"../2025-06-26-SpectractorExtraction-FromButler/data/missing_merged/exposurelist_and_auxtel_run_u_dagoret_auxtel_run_20250921_w_2025_36_spectractorv32_embargo_ptc_holoallfilt_a_v1_merged.npy",
    "run_v6":"../2025-06-26-SpectractorExtraction-FromButler/data/missing_merged/exposurelist_and_auxtel_run_u_dagoret_auxtel_run_20250921_w_2025_38_spectractorv32_main_gains_holoallfilt_a_v1_merged.npy",
    "run_7":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro_joined/auxtel_run_v3_2_0_repo_main_w_2025_38_gain-join-v3_2_0_repo_embargo_w_2025_36_ptc.npy",
    "run_v8": "None",
    "run_v9": "None",
    "run_v10": "None",
    "run_v11": "None",
    "run_v12": "None",
    "run2026_v01" : "None",
}

def DumpConfig():
    """
    Dump the configuration chosen here
    """
    print("**************************************************************************")
    print("*                         DumpConfig()                                   *")
    print(f" \t - version_run = {version_run}")
    if FLAG_REPO_EMBARGO:
        print(f" \t - butler repo : /repo/embargo")
    else:
        print(f" \t - butler repo : /repo/main")
    print(f" \t - annotation tag = {legendtag[version_run]}")
    print(f" \t - butler_collection = {butlerusercollectiondict[version_run]}")
    print(f" \t - extracted file from butler collection = {extractedfilesdict[version_run]}")
    print(f" \t - extracted and merged file from butler collection = {mergedextractedfilesdict[version_run]}")
    print(f" \t - mergedtofindmissings file from butler collection = {mergedtofindmissings[version_run]}")
    print("**************************************************************************")
    
    
          