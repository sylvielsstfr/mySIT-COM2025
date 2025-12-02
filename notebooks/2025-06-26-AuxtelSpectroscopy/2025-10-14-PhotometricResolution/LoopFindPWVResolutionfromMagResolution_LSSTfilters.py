# Find Resolution on PWV to achieve a resolution on Magnitude
#- author Sylvie Dagoret-Campagne
#- affiliation IJCLab
#- creation date : 2025/10/28
#- last update : 2025/10/29
#- last update : 2025/11/04 : double number of PWV points



import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import pandas as pd

import matplotlib.ticker                         # here's where the formatter is
import os,sys
import re
import pandas as pd

from astropy.io import fits
from astropy import units as u
from astropy import constants as c

import yaml

from importlib.metadata import version
the_ver = version('getObsAtmo')
print(f"Version of getObsAtmo : {the_ver}")


sys.path.append('../lib')


plt.rcParams["figure.figsize"] = (8,6)
plt.rcParams["axes.labelsize"] = 'xx-large'
plt.rcParams['axes.titlesize'] = 'xx-large'
plt.rcParams['xtick.labelsize']= 'xx-large'
plt.rcParams['ytick.labelsize']= 'xx-large'

props = dict(boxstyle='round', facecolor='white', alpha=0.5)


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)


from libPhotometricCorrections import *

from getObsAtmo import ObsAtmo


from libPhotometricCorrections import *

from rubinsimphot.phot_utils import Bandpass, Sed
from rubinsimphot.data import get_data_dir




def GenerateMultiValues(mean,sigma,size, lognorm_flag=True):
    """
    """
    if lognorm_flag:
        mu = np.log(mean**2/np.sqrt(mean**2+sigma**2))
        sig = np.sqrt(np.log(1+sigma**2/mean**2))
        all_values = np.random.lognormal(mean=mu, sigma=sig,size=size)
    else:
        mu = mean
        sig = sigma
        all_values = np.random.normal(mu,sig,size=size)
        
    return all_values


def CalculateMagsAndMagResolutions(pwv_values, pc,the_sed):
    """
    """

    # compute standard magnitude form the average called std
    mag_std = {}
    adu_std = {}
    atm_bands = pc.bandpass_total_std
    for index,f in enumerate(filter_tagnames) :
        mag_std[f] = the_sed.calc_mag(atm_bands[f])
        adu_std[f] = -2.5*np.log10(the_sed.calc_adu(atm_bands[f],photoparams))

    
    # magnitudes from non stadard pwv
    df = pd.DataFrame(columns = ["pwv","magu","magg","magr","magi","magz","magy","aduu","adug","adur","adui","aduz","aduy"])

    ## loop on pwv values
    for idx_pwv,pwv in enumerate(pwv_values):
        mag_nonstd = {}
        adu_nonstd = {}
        atm_bands = pc.coll_bandpass_total_nonstd[idx_pwv] 

        
        for index,f in enumerate(filter_tagnames) :
            mag_nonstd[f] = the_sed.calc_mag(atm_bands[f])
            adu_nonstd[f] = -2.5*np.log10(the_sed.calc_adu(atm_bands[f],photoparams))

        # add this entry
        df.loc[idx_pwv] = [pwv, mag_nonstd["u"],mag_nonstd["g"],mag_nonstd["r"],mag_nonstd["i"],mag_nonstd["z"],mag_nonstd["y"],
                       adu_nonstd["u"],adu_nonstd["g"],adu_nonstd["r"],adu_nonstd["i"],adu_nonstd["z"],adu_nonstd["y"]] 

    df = df[["pwv","aduu","adug","adur","adui","aduz","aduy"]]

    # compute the magnitude difference
    for index,f in enumerate(filter_tagnames) :
        label_in = f'adu{f}'
        label_out =f'd_adu{f}'
        df[label_out] = (df[label_in]- adu_std[f])*1000. 

    # Drop absolute mags and keep mag difference
    df = df.drop(labels=["aduu","adug","adur","adui","aduz","aduy"],axis=1)


    #compute relative color difference
    df["d_R-I"] = df["d_adur"] -  df["d_adui"]
    df["d_I-Z"] = df["d_adui"] -  df["d_aduz"]
    df["d_Z-Y"] = df["d_aduz"] -  df["d_aduy"]

    return df 



def GetdPWVvsPWV_FromMagResolution(all_PWV_values,all_DPWV_values, magresocut=5.0):
    """
    """
    sel_dpwv = []
    sel_df = []
    sel_pwv = []

    # loop on PWV values
    for pwv0 in all_PWV_values:

        # initialize atmosphere for the typical average conditions pwv0
        pc = PhotometricCorrections(am0,pwv0,oz0,tau0,beta0)


        # create a flat SED and noramize it to have mag 20 in z
        the_sed_flat = Sed()
        the_sed_flat.set_flat_sed()
        the_sed_flat.name = 'flat'
        zmag = 20.0
        flux_norm = the_sed_flat.calc_flux_norm(zmag, pc.bandpass_total_std['z'])
        the_sed_flat.multiply_flux_norm(flux_norm)
    
        # loop on PWV resolution
        for dpwv in all_DPWV_values:
            if dpwv >= pwv0:
                continue
                
            # compute the subsamples with varying PWV
            pwv_samples = GenerateMultiValues(pwv0,dpwv, NSAMPLES,lognorm_flag=True)
            pwv_samples= pwv_samples[np.where(np.logical_and(pwv_samples>0., pwv_samples<20.))[0]]
           
            pc.CalculateMultiObs(am0,pwv_samples,oz0,tau0,beta0)

            #compute distribution for magnitude resolution
            df = CalculateMagsAndMagResolutions(pwv_samples, pc, the_sed_flat)
            df_stat = df.describe()
            rms_y = df_stat.loc["std"]["d_aduy"]
            rms_zy = df_stat.loc["std"]["d_Z-Y"]

            if rms_y <=  magresocut: 
                print(f"pwv0 = {pwv0:.3f} mm , dpwv = {dpwv:.3f} mm , rms_y = {rms_y:.2f} mmag rms_z-y = {rms_zy:.2f} mmag")
                sel_dpwv.append(dpwv)
                sel_df.append(df)
                sel_pwv.append(pwv0)
                break
                
    return np.array(sel_pwv), np.array(sel_dpwv), sel_df
        
def set_photometric_parameters(exptime, nexp, readnoise=None):
    # readnoise = None will use the default (8.8 e/pixel). Readnoise should be in electrons/pixel.
    photParams = PhotometricParameters(exptime=exptime, nexp=nexp, readnoise=readnoise)
    return photParams

def scale_sed(ref_mag, ref_filter, sed):
    fluxNorm = sed.calc_flux_norm(ref_mag, lsst_std[ref_filter])
    sed.multiply_flux_norm(fluxNorm)
    return sed








## START HERE

### Configuration

emul = ObsAtmo("LSST")

# reference flux in Jy
F0 = ((0.*u.ABmag).to(u.Jy)).value
#print(F0)



# set default photometric parameters to compute ADU
photoparams = set_photometric_parameters(30, 1 , readnoise=None)


am0 = 2.0    # airmass
am_num = int(am0*10.)
pwv0 = 4.0  # Precipitable water vapor vertical column depth in mm
oz0 = 300.  # Ozone vertical column depth in Dobson Unit (DU)
ncomp=1     # Number of aerosol components
tau0= 0.0 # Vertical Aerosol depth (VAOD) 
beta0 = 1.2 # Aerosol Angstrom exponent


pc = PhotometricCorrections(am0,pwv0,oz0,tau0,beta0)

fig, axs = plt.subplots(1,1,figsize=(6,4))
axs.plot(pc.WL,pc.atm_std,'k-')
axs.set_xlabel("$\\lambda$ (nm)")
axs.set_title("Standard atmosphere transmission")
plt.show() 


fig, axs = plt.subplots(1,1,figsize=(6,4))
# loop on filter
for index,f in enumerate(filter_tagnames):
    
    axs.plot(pc.bandpass_inst[f].wavelen,pc.bandpass_inst[f].sb,color=filter_color[index]) 
    axs.fill_between(pc.bandpass_inst[f].wavelen,pc.bandpass_inst[f].sb,color=filter_color[index],alpha=0.2) 
    axs.axvline(FILTERWL[index,2],color=filter_color[index],linestyle="-.")
    
axs.set_xlabel("$\\lambda$ (nm)")
axs.set_title("Instrument throughput (auxtel)")
plt.show() 


fig, axs = plt.subplots(1,1,figsize=(6,4))
# loop on filter
for index,f in enumerate(filter_tagnames):
    
    axs.plot(pc.bandpass_total_std[f].wavelen,pc.bandpass_total_std[f].sb,color=filter_color[index]) 
    axs.fill_between(pc.bandpass_total_std[f].wavelen,pc.bandpass_total_std[f].sb,color=filter_color[index],alpha=0.2) 
    axs.axvline(FILTERWL[index,2],color=filter_color[index],linestyle="-.")
    
axs.set_xlabel("$\\lambda$ (nm)")
axs.set_title("Total filter throughput (auxtel)")
plt.show() 


# Find the throughputs directory 
#fdir = os.getenv('RUBIN_SIM_DATA_DIR')
fdir = get_data_dir()
if fdir is None:  #environment variable not set
    fdir = os.path.join(os.getenv('HOME'), 'rubin_sim_data')


the_sed_flat = Sed()
the_sed_flat.set_flat_sed()
the_sed_flat.name = 'flat'
zmag = 15.0
flux_norm = the_sed_flat.calc_flux_norm(zmag, pc.bandpass_total_std['z'])
the_sed_flat.multiply_flux_norm(flux_norm)


fig,ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(the_sed_flat .wavelen,-2.5*np.log10(the_sed_flat.fnu/F0),"b-",label=the_sed_flat.name)

ax.legend()
#ax.set_ylim(1e-17,1e-14)
#ax.set_xlim(300.,2000.)
ax.set_title("Flat SED $F_\\nu$")
ax.set_ylabel(" Magnitude = $-2.5 \log_{10}(F_\\nu/F_0)$")
ax.set_xlabel("$\\lambda \, (nm)$")
ax.yaxis.set_inverted(True)


ax3 = ax.twinx()
for ifilt,f in enumerate(filter_tagnames):
    ax3.fill_between(pc.bandpass_total_std[f].wavelen,pc.bandpass_total_std[f].sb,color=filter_color[ifilt],alpha=0.1) 
    ax3.set_yticks([])



fig,ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(the_sed_flat .wavelen,-2.5*np.log10(the_sed_flat.fnu/F0),"b-",label=the_sed_flat.name)

ax.legend()
#ax.set_ylim(1e-17,1e-14)
#ax.set_xlim(300.,2000.)
ax.set_title("Flat SED $F_\\nu$")
ax.set_ylabel(" Magnitude = $-2.5 \log_{10}(F_\\nu/F_0)$")
ax.set_xlabel("$\\lambda \, (nm)$")
ax.yaxis.set_inverted(True)


ax3 = ax.twinx()
for ifilt,f in enumerate(filter_tagnames):
    ax3.fill_between(pc.bandpass_total_std[f].wavelen,pc.bandpass_total_std[f].sb,color=filter_color[ifilt],alpha=0.1) 
    ax3.set_yticks([])
plt.show()


# LOOP on main PWV then on PWV resolution


NSAMPLES = 1000

# PWV values from 1 to 20 mm
all_PWV = np.arange(1,20.,0.5)
#print(all_PWV) 


#
MAGRESOCUT = 10.0 # mmag 
all_DPWV = np.arange(2.,0.,-0.02)
#print(all_DPWV)
sel_pwv0, sel_dpwv0,sel_df0 = GetdPWVvsPWV_FromMagResolution(all_PWV,all_DPWV, magresocut=MAGRESOCUT)


MAGRESOCUT = 5.0 # mmag 
all_DPWV = np.arange(1.,0.,-0.015)
sel_pwv1, sel_dpwv1,sel_df1 = GetdPWVvsPWV_FromMagResolution(all_PWV,all_DPWV, magresocut=MAGRESOCUT)


MAGRESOCUT = 1.0 # mmag 
all_DPWV = np.arange(0.5,0.,-0.01)
sel_pwv2, sel_dpwv2,sel_df2 = GetdPWVvsPWV_FromMagResolution(all_PWV,all_DPWV, magresocut = MAGRESOCUT)



data = {
    "pwv10mmag" : sel_pwv0.tolist(),
    "dpwv10mmag" : sel_dpwv0.tolist(),
    "pwv05mmag" : sel_pwv1.tolist(),
    "dpwv05mmag" : sel_dpwv1.tolist(),
    "pwv01mmag" : sel_pwv2.tolist(),
    "dpwv01mmag" : sel_dpwv2.tolist()
}

with open(f"pwvdpwvdata_resomagY_airmass{am_num:0}.yaml", 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)






















