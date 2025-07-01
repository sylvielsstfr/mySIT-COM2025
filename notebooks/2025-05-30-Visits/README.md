# README.md

- author : Sylvie Dagoret-Campagne
- affiliation : IJCLab/IN2P3/CNRS
- member : DESC, rubin-inkind
- creation date : 2025-06-01
- last update : 2025-06-06 : psf and Ap fluxes
## notebooks

- **01_FindObservationsInButlerRegistry.ipynb** : Extract from butler registry the series of visiits including the information that comes with it.
  	       
- **02_FindObservationsInButlerRegistryInTractPatch.ipynb** : As above but allocate tract and patch at each visit.
 
- **03_histosforvisits.ipynb** : Make plots from the visits to find the most visited tract and patches.

- **04_DeepCoaddFromSelectedTractPatch.ipynb** : Show the DeepCoadd of the most visited tract and patch.

- **05_SourcesFromVisits.ipynb** : Making Light curves from objects found in the DeepCoadds (**04_DeepCoaddFromSelectedTractPatch.ipynb**)


- **06_SourcesFromVisitsMultiBands_psfMag.ipynb** : Light curves on psfFlux

- **07_SourcesFromVisitsMultiBands_apMag.ipynb** : Light curves on psfFlux, calibFlux and all Ap fluxes. Generate the resolution file for the next notebook.

- **07b_CheckFluxesCalibrations.ipynb** : Check the calibration of aperture fluxes and which is the best radius for the aperture flux for stars.
