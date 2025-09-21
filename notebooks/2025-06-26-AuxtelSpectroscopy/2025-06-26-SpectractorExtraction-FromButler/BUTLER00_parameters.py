# input file configuration


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