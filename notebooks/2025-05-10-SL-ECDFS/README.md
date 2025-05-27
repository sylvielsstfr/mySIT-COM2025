# README.md : Commissioning with ComCam on Strong Lenses in ECDFS

- author : Sylvie Dagoret-Campagne
- affiliation : IJCLab.in2p3.fr
- creation date : 2025-05-27



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
- Forced Photometry Object from DRP analysis with PSF fluxes

## SL08_ManyLightCurvesOneTargetLSSTComCamDRPSourcesSersicprof.ipynb
- Forced Photometry Object from DRP analysis with other fluxes

## SL09_LuptonColoredCutouts.ipynb
- Colored images wit make_lupton

