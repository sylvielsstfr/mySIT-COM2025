#README.md

- author : Sylvie Dagoret-Campagne
- affiliation: IJCLab/IN2P3/CNRS
- creation date : 2025-05-23
- last update : 2025-06-22


base                   /Users/dagoret/miniconda3
conda_py311            /Users/dagoret/miniconda3/envs/conda_py311
conda_py311_rail       /Users/dagoret/miniconda3/envs/conda_py311_rail
conda_py313          * /Users/dagoret/miniconda3/envs/conda_py313
conda_py313-hyrax      /Users/dagoret/miniconda3/envs/conda_py313-hyrax
conda_py313_jax        /Users/dagoret/miniconda3/envs/conda_py313_jax
conda_py313_rail       /Users/dagoret/miniconda3/envs/conda_py313_rail



## Make cutout from GEMS (HST) data

- **HSTGEMS01_ExtractCutoutGEMSHST.ipynb**: Make a cutout in 200 x 200 pixel size. Need to specify the target SL and choose the HST V or Z bands used by the GEMS survey      

- **HSTGEMS01b_ExtractCutoutGEMSHST.ipynb**: Same things

-     
## Find the tile number corresponding to a ra,dec coordinates- 
- **HSTGEMS02_FindTileForCutoutGEMSHST.ipynb**: : Done because one tile number was wrong in the reference article


## View the cutout generated in **HSTGEMS01_ExtractCutoutGEMSHST.ipynb**:    
- **HSTGEMS03_ViewCutouts.ipynb**

## View cutout , lupton colored cutouts and corlor diff image
- **HSTGEMS04_ViewCutouts-andMakeLupton.ipynb**

## Loop on ## Viewing cutout , lupton colored cutouts and corlor diff image  
- **HSTGEMS05_LoopOnViewCutouts-andMakeLupton.ipynb**
- 

## Study Flux and Weight :
- **HSTGEMS06_CompareFluxAndWeights.ipynb** 

## Start to Fit SL with lenstronomy

- **HSTGEMS07_FitLensWithLenstronomy.ipynb** : stable version
- **HSTGEMS07_FitLensWithLenstronomy-dev.ipynb** : developpement version

- **HSTGEMS07_FitLensWithLenstronomy_ECDFS_G15422.ipynb**  : version for  ECDFS_G15422                      
## Start to Fit SL with jaxstronomy

- **HSTGEMS08_FitLensWithJaxtronomy_ECDFS_G15422.ipynb**  
                       
        



## In construction for elsewhere
SL09_LuptonColoredCutouts.ipynb



