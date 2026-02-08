# README.md
- author : Sylvie Dagoret-Campagne
- creation date : 2026-01-16
- last update : 2026-01-28
- last update : 2026-02-07

## Detect which targets have bad DCCD and chi2_fit parameters:

- **QCUT01_ExploreHoloQuality.ipynb**:

## Access to Spectra in Butler:

- **QCUT02_CheckFitQualityInSpectractorResults.ipynb**:

## Normalise the chi2 to compare between targets

- **QCUT03_QualityCutPerTarget.ipynb**:                          
- **QCUT03b_QualityCutPerTarget.ipynb**:   

## Compare chi2 wrt telescope parameters

- **QCUT04_ExploreHoloQuality_ImpactOfExternalConditions.ipynb**:  

## Compare chi2 wrt weather parameters

- **QCUT05_ExploreHoloQuality_ImpactOfWeatherConditions.ipynb**: 

## Use ConstDB to access to exp_time vs time

- **QCUT06_ConstDB_target38666.ipynb**:
- **QCUT07_ConstDB_ExternalParameters.ipynb**:

## Parameters histograms with pdf files written
- **QCUT08_ExploreHoloQuality_ImpactOfAllQualityCuts.ipynb** : histos and 2D plots on parameters : no cut applied and write accuracy and resolution on PWV in csv file (when no cut is applied)

- **QCUT09** : More ranges in parameter
- **QCUT09_ExploreHoloQuality_ImpactOfAllQualityCuts_bytarget.ipynb**
- **QCUT09c_ExploreHoloQuality_ImpactOfLessQualityCuts_byGaiatarget.ipynb** : histograms focused on Gaia

## Generate config files for cuts in json file
- **QCUT10_SelectQualityCuts-GenerateDefaultConfig.ipynb**:

## Apply per User cuts on accuracy-resolution PWV plot per filter and generate csv table output
- **QCUT11_SelectQualityCuts-CompareUsersConfig.ipynb**

## Combine/compare no-cut (QCUT08) / user applied cuts (QCUT11) in tables (reading csv files written) 
- **QCUT12_SummaryQualityCuts-ComparePWV_accuracy_resolution_Users.ipynb**

