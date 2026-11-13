# input file configuration
# author : Sylvie Dagoret-Campagne
# creation date : 2026-11-13

import pprint

# Select run version tag to be used in EXTR_viewSpectractor notebooks 
#version_run = "run2026_v01"
#version_run = "run2026_v02d_cr"
version_run = "run2023-2026_JanToJul_cr"
# Configuration for the butler repo associated to the version_run

# Runs from Corentin Ravoux added 2026-02-18
#"run2026_v02b_cr :'auxtel_atmosphere_feb26_gaiaspec_calspecgaiatarget_calspecthroughput_preoct23.npy',
#"run2026_v02d_cr :'auxtel_atmosphere_feb26_gaiaspec_gaiatarget_calspecthroughput.npy',

# Runs from Corentin Ravoux generated in 2026
#auxtel_atmosphere_calspecgaiatarget_monthly_jan26.npy
#auxtel_atmosphere_calspecgaiatarget_monthly_feb26.npy
#auxtel_atmosphere_calspecgaiatarget_monthly_mar26.npy
#auxtel_atmosphere_calspecgaiatarget_monthly_apr26.npy
#auxtel_atmosphere_calspecgaiatarget_monthly_mai26.npy
#auxtel_atmosphere_calspecgaiatarget_monthly_jun26.npy
#auxtel_atmosphere_calspecgaiatarget_monthly_jul26.npy





map_run_butler_embargo = { 
                            "run2026_v02b_cr": False,
                            "run2026_v02d_cr": False,
                            "run2026_mm01_cr": False,
                            "run2026_mm02_cr": False, 
                            "run2026_mm03_cr": False, 
                            "run2026_mm04_cr": False,
                            "run2026_mm05_cr": False, 
                            "run2026_mm06_cr": False, 
                            "run2026_mm07_cr": False, 
                            "run2026_v02b_cr_run2026_v02d_cr" : False,
                            "run2026_JanToJul_cr" : False,
                            "run2023-2026_JanToJul_cr": False,
                         }

FLAG_REPO_EMBARGO = map_run_butler_embargo[version_run]

# Associate the tag to the Spectractor runparameters (to be used in plots)
legendtag = {
"run2026_v02b_cr" : "cr_feb26_gaiaspec_calspecgaiatarget_calspecthroughput_preoct23",
"run2026_v02d_cr" : "cr_feb26_gaiaspec_gaiatarget_calspecthroughput",
"run2026_v02b_cr_run2026_v02d_cr" : "concat(cr_feb26_gaiaspec_calspecgaiatarget_calspecthroughput_preoct23,cr_feb26_gaiaspec_gaiatarget_calspecthroughput)",
"run2026_mm01_cr" : "cr_run2026-Jan", 
"run2026_mm02_cr" : "cr_run2026-Feb", 
"run2026_mm03_cr" : "cr_run2026-Mar", 
"run2026_mm04_cr" : "cr_run2026-Apr", 
"run2026_mm05_cr" : "cr_run2026-May", 
"run2026_mm06_cr" : "cr_run2026-Jun", 
"run2026_mm07_cr" : "cr_run2026-Jul", 
"run2026_JanToJul_cr" :  "cr_runs2026-Jan-Jul",
"run2023-2026_JanToJul_cr" : "cr_run2023-2026_JanToJul",
}



# List of user collection in butler  where the results of spectractor run are
butlerusercollectiondict = {
    "run2026_v02b_cr" : "cr_feb26_gaiaspec_calspecgaiatarget_calspecthroughput_preoct23",
    "run2026_v02d_cr" : "cr_feb26_gaiaspec_gaiatarget_calspecthroughput",
    "run2026_v02b_cr_run2026_v02d_cr" : "cr_feb26_gaiaspec_calspecgaiatarget_calspecthroughput_preoct23-cr_feb26_gaiaspec_gaiatarget_calspecthroughput",
}


# path of output files Spectractor parameters Extracted from Butler 
extractedfilesdict = {

"run2026_v02b_cr" : "../2026-11-13-SpectractorExtraction-FromButler/data/spectro_from_corentin/auxtel_atmosphere_feb26_gaiaspec_calspecgaiatarget_calspecthroughput_preoct23.npy",

"run2026_v02d_cr" : "../2026-11-13-SpectractorExtraction-FromButler/data/spectro_from_corentin/auxtel_atmosphere_feb26_gaiaspec_gaiatarget_calspecthroughput.npy",

"run2026_m01_cr" : "../2026-11-13-SpectractorExtraction-FromButler/data/spectro_from_corentin/auxtel_atmosphere_calspecgaiatarget_monthly_jan26.npy",
    
"run2026_m02_cr" : "../2026-11-13-SpectractorExtraction-FromButler/data/spectro_from_corentin/auxtel_atmosphere_calspecgaiatarget_monthly_feb26.npy",
    
"run2026_m03_cr" : "../2026-11-13-SpectractorExtraction-FromButler/data/spectro_from_corentin/auxtel_atmosphere_calspecgaiatarget_monthly_mar26.npy",

"run2026_m04_cr" : "../2026-11-13-SpectractorExtraction-FromButler/data/spectro_from_corentin/auxtel_atmosphere_calspecgaiatarget_monthly_apr26.npy",

"run2026_m05_cr" : "../2026-11-13-SpectractorExtraction-FromButler/data/spectro_from_corentin/auxtel_atmosphere_calspecgaiatarget_monthly_mai26.npy",

"run2026_m06_cr" : "../2026-11-13-SpectractorExtraction-FromButler/data/spectro_from_corentin/auxtel_atmosphere_calspecgaiatarget_monthly_jun26.npy",

"run2026_m07_cr" : "../2026-11-13-SpectractorExtraction-FromButler/data/spectro_from_corentin/auxtel_atmosphere_calspecgaiatarget_monthly_jul26.npy",
}

merge_composition = {
"run2026_v02b_cr_run2026_v02d_cr" : [ "run2026_v02b_cr", 
                                      "run2026_v02d_cr"],
    
"run2026_JanToJul_cr" :  ["run2026_m01_cr",
                       "run2026_m02_cr",
                       "run2026_m03_cr",
                       "run2026_m04_cr",
                       "run2026_m05_cr",
                       "run2026_m06_cr",
                       "run2026_m07_cr"],   
"run2023-2026_JanToJul_cr" : [ "run2026_v02b_cr", 
                            "run2026_v02d_cr",
                            "run2026_m01_cr",
                            "run2026_m02_cr",
                            "run2026_m03_cr",
                            "run2026_m04_cr",
                            "run2026_m05_cr",
                            "run2026_m06_cr",
                            "run2026_m07_cr"], 
}

# path of output files Spectractor parameters Extracted from Butler and merged with exposure list from butler registry
mergedextractedfilesdict = {
"run2026_v02b_cr_run2026_v02d_cr" : "../2026-11-13-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_atmosphere_feb26_gaiaspec_run2026_v02b_cr_run2026_v02d_cr_merged.npy", 
 "run2026_JanToJul_cr" :  "2026-11-13-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_atmosphere_calspecgaiatarget_run2026_JanToJul_cr_merged.npy", 
 "run2023-2026_JanToJul_cr" :  "2026-11-13-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_atmosphere_calspecgaiatarget_run2023-2026_JanToJul_cr_merged.npy", 
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
    #print(f" \t - butler_collection = {butlerusercollectiondict[version_run]}")
    if version_run in extractedfilesdict:
        print(f" \t - Extracted file from butler collection = {extractedfilesdict[version_run]}")
    if version_run in mergedextractedfilesdict:    
        print(f" \t - Merged file from butler collection = {mergedextractedfilesdict[version_run]}")
    print("\t - Merge Composition ::")
    pprint.pprint(merge_composition)    
    print("**************************************************************************")
    
    
          