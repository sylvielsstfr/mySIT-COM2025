# README.md

- creation : 2025-09-21
- last update : 2025-10-21 run_v9
- last update : 2025-10-23 run_v10 

## Tools

- **TOOL_QueryCollectionsAndDatasetsInButler.ipynb**: Search for relevant collection 

- **TOOL_ListOfExposures-hologram-repomain-new25-09.ipynb** : Extract exposures from Butler and generate a file

- **CHECK_ExposuresList.ipynb** : View the list of exposures file  extracted from butler

## Extractor Of Spectractor Results:
This is the first step to extract from Butler the atmospheric parameters and other quantities.

- **EXTR_viewSpectractorResults_September2025_repomain.ipynb** : Extract Spectractor data from butler and write them in a *.npy file

## Post processing

- **MERGE_pectractorResults_ExposuresList.ipynb**: merge Spectractor Results and Exposure list.



## Work to find missing Exposures
- 1) run **MERGE_ExposureList_SpectractorResults-FindMissing.ipyng*** to produce the list of exposure in butler associated with existing or not results in spectractor
     
- 2) run **CHECK_SpectractionEfficiency.ipynb** to generate barplot and csv file to provide the list of  missings and not missing (OK) results.
 




## Data
- **BUTLER00_parameters.py** contain path to files

       version_run = "run_v3"
       legendtag = {"run_v1" : "v3.1.0","run_v2":"v3.1.0","run_v3":"v3.2.0", "run_v4":"v3.2.0"}

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
        }



        extractedfilesdict = {
        # /repo/main
        "run_v1": "../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_atmosphere_20250912a_repomain_v1.npy",
        "run_v2": "../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_20250917_w_2025_25_spectractorv31_holoallfilt_a_repomain_v1.npy",
        # /repo/embargo, gain
        "run_v4":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_u_dagoret_auxtel_run_20250920_w_2025_36_spectractorv32_embargo_gains_holoallfilt_b_v1.npy",
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




