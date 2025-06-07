# README.md : Commissioning with ComCam on Strong Lenses in ECDFS

- author : Sylvie Dagoret-Campagne
- affiliation : IJCLab.in2p3.fr
- creation date : 2025-05-27
- last update : 2025-06-07


## SL00_StrongLensSkymapInTractPatch.ipynb
- Location of SL in Tract

## SL01_LSSTComCamDeepCoadds.ipynb
- show the whole deep coadds in the 6 bands (u,g,r,i,z,y) where the SL are located.


## SL02_SearchForObjetcsCutouts.ipynb
- show cutout from Deep Coadds in the 6 bands (u,g,r,i,z,y) around SL 
- View cutouts GEMS SL in ECDFS from Deep Coadds in the 6 bands (u,g,r,i,z,y) from LSSTComCam
- And save them in a fits file including their image, variance plane, mask and WCS. Moreover the psf is save in another file. Note the variance is saved in a sparate file. However it is not necessary as the variance is also saved in the wcs file (rubin-format).

## SL03_SearchForObjectInTables.ipynb
- Find the closest Rubin object in Object Table. Be carefull Do not force parentObjectId=0 otherwise the SL will not be found !!!. Shows the separation angle and compate the object in the objectTable with the known position of the SL

## SL04_LSSTComCamOneTargetDiaObjectsAndCutouts.ipynb
- Compare SL position with DIA object position. Almot similar to SL03 except with DIA objects.


## SL05_ManyLightCurvesOneTargetLSSTComCamDiaSourcesForcedPhoto.ipynb
- Light curves using DIA Fourced sources ==> OK but creazy variations

## SL06_ManyLightCurvesOneTargetLSSTComCamDiaSources.ipynb
- Light curves using only DIA sources ==> Not enough DIA source in the SL position

## SL07_ManyLightCurvesOneTargetLSSTComCamDRPSourcesPSFprof.ipynb
- Light curves from forcedSourceTable (Forced Photometry Object from DRP analysis) with PSF fluxes.
- Lighght curves are also colored by airmass and azimuth extraction from visit and sky background level.
- 
## SL07b_ManyLightCurvesOneTargetLSSTComCamDRPSourcesPSFprof.ipynb
- Light curves from **mergedforcedSourceTable** (Forced Photometry Object from DRP analysis) with PSF fluxes.
- Lighght curves are also colored by airmass and azimuth extraction from visit and sky background level.
- Deprecated because **mergedforcedSourceTable** does not bring anything more than **SL07_ManyLightCurvesOneTargetLSSTComCamDRPSourcesPSFprof.ipynb**


## SL07c_LooponSL_ManyLightCurvesOneTargetLSSTComCamDRPSourcesPSFprof.ipynb:
- Loop on SL to show all light curves (does what **SL07_ManyLightCurvesOneTargetLSSTComCamDRPSourcesPSFprof.ipynb** does)

## SL08_ManyLightCurvesOneTargetLSSTComCamDRPSourcesSersicprof.ipynb
- Light curves from forcedSourceTable (Forced Photometry Object from DRP analysis) with other fluxes than PSF fluxes : However those flux are not available
- - Deprecated because no other profile are available for the moments

## SL09_LuptonColoredCutouts.ipynb
- Colored images wit make_lupton

