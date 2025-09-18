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

    