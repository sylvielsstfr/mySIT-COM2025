# README.md

- author : Sylvie Dagoret-Camapgne
- creation date : 2025-02-27
- last update : 2025-03-09

## Try Periodogram

- **PeriodogramAtmosphereFrmMerra2_timeseq.ipynb**

## Extract TimeScales from FFT 

- **FFTAtmosphFromMerra2_timeseq.ipynb** 

## Fit with Gaussian Process

### Fit periodic components (2025-02-26)

- **FitGPPeriodicAtmMerra2_timeseq.ipynb**: Fit the periodic components using Gaussian Processes with the periodic kernel. We fit the amplitude after selecting a sumbsample of the Merra2 data. The Gaussian process fit cannot be performed on the 30000 Merra2 samples.

### Fit random component (2025-02-27)

- **FitGPShortTimeScaleAtmMerra2_timeseq.ipynb** : must run **FitGPPeriodicAtmMerra2_timeseq.ipynb** before in order to generate residuals time sequences in Forlder dataFitGPPerAtmosphereFomMerra2.
- After subtraction of the periodic component, we fit a anrdom gaussian process to describe the irregularities of the variations.


### Method LombScargle (2025-02-27)

Replace FFT by Lomb-Scargle because there are missing values:
- **LombScargleAtmosphFromMerra2_timeseq.ipynb**: https://docs.astropy.org/en/latest/timeseries/lombscargle.html
Very usefull to Auxtel data

### DCF (2025-03-08)

- **DCTAtmosphFromMerra2_timeseq.ipynb**
- **DCTonResidualsFromMerra2_timeseq.ipynb**



## Deprecated
- FitAtmosphereFrmMerra2_timeseq-v2.ipynb
- FitAtmosphereFrmMerra2_timeseq-v1.ipynb
- FitGPPeriodicShortTimeScaleAtmMerra2_timeseq.ipynb to be removed



               



