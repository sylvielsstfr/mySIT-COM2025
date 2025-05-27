# README.md : Commissioning with ComCam on Strong Lenses in ECDFS

- author : Sylvie Dagoret-Campagne
- affiliation : IJCLab.in2p3.fr
- creation date : 2025-05-27



## SL01_LSSTComCamDeepCoadds.ipynb
- show the whole deep coadds in the 6 bands (u,g,r,i,z,y) where the SL are located.


## SL02_SearchForObjetcsCutouts.ipynb
- show cutout from Deep Coadds in the 6 bands (u,g,r,i,z,y) around SL 
- View cutouts GEMS SL in ECDFS from Deep Coadds in the 6 bands (u,g,r,i,z,y) from LSSTComCam
- And save them in a fits file including their image, variance plane, mask and WCS. Moreover the psf is save in another file.

## SL03_SearchForObjectInTables.ipynb
- Find the closest Rubin object in Object table --> Not sure it works, why ?

## SL04_LSSTComCamOneTargetDiaObjectsAndCutouts.ipynb
- Compare SL position with DIA object position


## SL05_ManyLightCurvesOneTargetLSSTComCamDiaSourcesForcedPhoto.ipynb
- Light curves using DIA Fourced sources ==> OK but creazy variations

## SL06_ManyLightCurvesOneTargetLSSTComCamDiaSources.ipynb
- Light curves using only DIA sources ==> Not enough DIA source in the SL position

## SL07_ManyLightCurvesOneTargetLSSTComCamDRPSourcesPSFprof.ipynb
- Forced Photometry Object from DRP analysis

## SL08_ManyLightCurvesOneTargetLSSTComCamDRPSourcesSersicprof.ipynb
