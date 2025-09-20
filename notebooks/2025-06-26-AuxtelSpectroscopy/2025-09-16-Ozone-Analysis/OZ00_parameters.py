# input file configuration

version_results = "v3"
legendtag = {"v1" : "v3.1.0","v2":"v3.1.0", "v3":"v3.1.0"}

atmfilenamesdict = {"v1":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_atmosphere_20250912a_repomain_v1.npy",
                    "v2":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro/auxtel_run_20250917_w_2025_25_spectractorv31_holoallfilt_a_repomain_v1.npy",
                   "v3":"../2025-06-26-SpectractorExtraction-FromButler/data/spectro_merged/auxtel_run_20250917_w_2025_25_spectractorv31_holoallfilt_a_repomain_v1_merged.npy",}


atmfilename = atmfilenamesdict[version_results]
tag = legendtag[version_results] 

# Selection parameters for cuts
DCCDMINFIG = 185.0;
DCCDMAXFIG = 190.0;
DCCDMINCUT = 186.7;
DCCDMAXCUT = 188.0;
CHI2CUT = 300.;
EXPTIMECUT = 20.0;    
PWVMINCUT = 0.;
PWVMAXCUT = 20.;
OZMINCUT = 10.;
OZMAXCUT = 590.0;    

def getSelectionCutOld(df_spec, chi2max=20., pwvmin=0.1, pwvmax = 14.9):
    cut =  (df_spec["CHI2_FIT"]<chi2max) & (df_spec["PWV [mm]_x"] > pwvmin) & (df_spec["PWV [mm]_x"] < pwvmax) 
     #(df_spec["ozone [db]_y"] > ozmin) & (df_spec["ozone [db]_y"] < ozmax) & (df_spec["TARGET"] == "HD185975")
    return cut

    
def getSelectionCut(df_spec, chi2max=CHI2CUT, pwvmin=PWVMINCUT, pwvmax = PWVMAXCUT,ozmin=OZMINCUT,ozmax=OZMAXCUT):
    cut =  (df_spec["CHI2_FIT"]<chi2max) & (df_spec["ozone [db]_x"] > ozmin) & (df_spec["ozone [db]_x"] < ozmax) & (df_spec["D2CCD"]>DCCDMINCUT) &  (df_spec["D2CCD"]<DCCDMAXCUT) & \
    (df_spec['EXPTIME'] > EXPTIMECUT ) 
    return cut 

def getSelectionCutNoPolar(df_spec, chi2max=CHI2CUT, pwvmin=PWVMINCUT, pwvmax=PWVMAXCUT,ozmin=OZMINCUT,ozmax=OZMAXCUT):
    cut =  (df_spec["CHI2_FIT"]<chi2max) & (df_spec["ozone [db]_x"] > ozmin) & (df_spec["ozone [db]_x"] < ozmax) & (df_spec["D2CCD"]>DCCDMINCUT) &  (df_spec["D2CCD"]<DCCDMAXCUT) & \
    (df_spec['EXPTIME'] > EXPTIMECUT ) &   (df_spec["TARGET"] != "HD185975")
    return cut

def getSelectionCutWithPolar(df_spec, chi2max=CHI2CUT, pwvmin=PWVMINCUT, pwvmax=PWVMAXCUT,ozmin=OZMINCUT,ozmax=OZMAXCUT):
    cut =  (df_spec["CHI2_FIT"]<chi2max) & (df_spec["ozone [db]_x"] > ozmin) & (df_spec["ozone [db]_x"] < ozmax) & (df_spec["D2CCD"]>DCCDMINCUT) &  (df_spec["D2CCD"]<DCCDMAXCUT) & \
    (df_spec['EXPTIME'] > EXPTIMECUT ) &   (df_spec["TARGET"] == "HD185975")
    return cut

    


# Merra2 pathname

filename_m2 = "../2025-09-16-SpectroMerra2/MerradataMerged/Merge_inst1_2d_asm_Nx_M2I1NXASM-2021-2025.csv"
