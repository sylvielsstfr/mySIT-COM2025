# README.md

- creation : 2025-09-21
- last update : 2025-10-21 run_v9
- last update : 2025-10-23 run_v10
- last update : 2025-12-12 run v12

## Tools

- **TOOL_QueryCollectionsAndDatasetsInButler.ipynb**: Search for relevant collection 

- **TOOL_ListOfExposures-hologram-repomain-new25-09.ipynb** : Extract exposures from Butler and generate a file

- **CHECK_ExposuresList.ipynb** : View the list of exposures file  extracted from butler

## Extractor Of Spectractor Results:
This is the first step to extract from Butler the atmospheric parameters and other quantities.

- **EXTR_viewSpectractorResults_September2025_repomain.ipynb** : Extract Spectractor data from butler and write them in a *.npy file

- EXTR_viewSpectractorResults_December2025_repomain.ipynb : Extraction, old method
- **EXTR_viewSpectractorResults_December2025_repomain-new.ipynb** : Extraction (working 2025-12-12) with v12 (Spectractor test_ccdgains)


- **EXTR_viewSpectractorResults_December2025_repomain-new.py** : Extraction with scripts to be tested
- **EXTR_viewSpectractorResults_December2025_repomain.py**     : Extraction with scripts to be tested
- **EXTR_viewSpectractorResults_September2025_repomain.py**  : Extraction with scripts to be tested

## Post processing

- **MERGE_pectractorResults_ExposuresList.ipynb**: merge Spectractor Results and Exposure list.



## Work to find missing Exposures
- 1) run **MERGE_ExposureList_SpectractorResults-FindMissing.ipyng*** to produce the list of exposure in butler associated with existing or not results in spectractor
     
- 2) run **CHECK_SpectractionEfficiency.ipynb** to generate barplot and csv file to provide the list of  missings and not missing (OK) results.
 

## Statistics on Exposures and Spectra

Generate figures of statistics: 

- 1) **CHECK_ExplosuresList.ipynb**: on butler registry exposures
- 2) **CHECK_SpectractionStatistics.ipynb** : On extracted spectra data
  3) **CHECK_SpectractionStatistics.ipynb** : On merged and extracted spectra data


## Data
- **BUTLER00_parameters.py** contain path to files

       version_run = "run_v10"

        legendtag = {"run_v1" : "v3.1.0 (/repo/main, w_2025_25,empty,gain)","run_v2":"v3.1.0 (/repo/main, w_2025_25,all-filts,gain)","run_v3":"v3.2.0 (/repo/embargo, w_2025_36,gain),", "run_v4":"v3.2.0 (/repo/embargo,w_2025_36,gain)","run_v5":"v3.2.0  (/repo/embargo,w_2025_36,ptc)","run_v6":"v3.2.0  (/repo/main,w_2025_38,gain)","run_v7":"v3_2_0_repo_main_w_2025_38_gain-v3_2_0_repo_embargo_w_2025_36_ptc","run_v8": "v3_2_0_repo_main_w_2025_42_ptc", "run_v9" : "v3_2_0_repo_main_w_2025_42_gains","run_v10" : "v3_2_0_repo_main_w_2025_42_gains"}

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
    "run_v10":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_run_u_dagoret_auxtel_run_20251022_w_2025_42_spectractorv32_all_main_data_gains_holoallfilt_a_v1_merged.npy"
            }




## Directory Data

      $ tree data
      data
      |-- butlerregistry : Exposures from butler
      |   |-- holosummary_all_filters_repo_embargo.csv
      |   `-- holosummary_all_filters_repo_main.csv
      |-- spectro : file extracted by EXTR_viewSpectractorResults_September2025_repomain.ipynb
      |   |-- auxtel_atmosphere_20250625a_v1.npy
      |   |-- auxtel_atmosphere_20250702a_repomain_v1.npy
      |   |-- auxtel_atmosphere_20250702b_repomain_v1.npy
      |   |-- auxtel_atmosphere_20250703a_repomain_v1.npy
      |   |-- auxtel_atmosphere_20250912a_repomain_v1.csv
      |   |-- auxtel_atmosphere_20250912a_repomain_v1.hdf5
      |   |-- auxtel_atmosphere_20250912a_repomain_v1.npy
      |   |-- auxtel_atmosphere_20250912a_repomain_v1.parquet.gzip
      |   |-- auxtel_run_20250917_w_2025_25_spectractorv31_holoallfilt_a_repomain_v1.npy
      |   |-- auxtel_run_u_dagoret_auxtel_run_20250920_w_2025_36_spectractorv32_embargo_gains_holoallfilt_a_v1.npy
      |   |-- auxtel_run_u_dagoret_auxtel_run_20250920_w_2025_36_spectractorv32_embargo_gains_holoallfilt_b_v1.npy
      |-- spectro_merged : files merged with MERGE_pectractorResults_ExposuresList.ipynb
          |-- auxtel_run_20250917_w_2025_25_spectractorv31_holoallfilt_a_repomain_v1_merged.npy
          |-- auxtel_run_u_dagoret_auxtel_run_20250920_w_2025_36_spectractorv32_embargo_gains_holoallfilt_b_v1_merged.npy




