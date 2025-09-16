# Last update 2024-11-14

import lsst.daf.butler as dafButler
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

from getObsAtmo.getObsAtmo import ObsAtmo
from datetime import datetime
import h5py
import os

# Spectra

def select_files(butler,collection, where):
    """
    Select all records according the where clause
    """
    # datasetRefs = registry.queryDatasets(datasetType='spectractorSpectrum', collections=my_collection, where=where)
    #records = list(butler.registry.queryDimensionRecords('exposure', where=where))
    records = list(butler.registry.queryDimensionRecords('exposure', datasets='spectractorSpectrum', where=where,  collections=collection))
    records = sorted(records, key=lambda x: x.id, reverse=False)
    return records

def filter_data(butler,collection,dateobs,records, sigma_clip=3,remove_withfilters = True, plot_flag = False):  # pragma: no cover
    """
    Spectrum reconstruction Quality Selection
    """
    from scipy.stats import median_abs_deviation
    D = []
    chi2 = []
    dx = []
    amplitude = []
    regs = []
    times = []
    specs = []
    alpha_0_2 = []
    #parameters.VERBOSE = False
    #parameters.DEBUG = False
    spec_flagfilter = []
    for i, r in enumerate(records):
        times.append(r.day_obs)
        spec = butler.get('spectractorSpectrum', visit=r.id, collections=collection, detector=0, instrument='LATISS')
        spec.dataId = r.id
        spec.physical_filter = r.physical_filter

        if spec.x0[0] > 500: 
            continue
            
        flag_filter = spec.physical_filter == 'empty~holo4_003'
        spec_flagfilter.append(flag_filter)
        
        D.append(spec.header["D2CCD"])
        dx.append(spec.header["PIXSHIFT"])
        regs.append(np.log10(spec.header["PSF_REG"]))
        # what is amplitude[300:]
        amplitude.append(np.sum(np.abs(spec.data[300:])))
        # if "CHI2_FIT" in header:
        chi2.append(spec.header["CHI2_FIT"])
        specs.append(spec)
        p = butler.get('spectrumForwardModelFitParameters', visit=r.id, collections=collection, detector=0, instrument='LATISS')
        alpha_0_2.append(p.values[p.get_index("alpha_0_2")])
        #except:
        #    new_file_names.remove(name)
        #    print(f"fail to open {name}. len(file_names)={len(new_file_names)}")
    params = {'D2CCD': np.array(D),
              'dx': np.array(dx),
              'regs': np.array(regs),
              'chi2': np.array(chi2),
              'amplitude': np.array(amplitude),
              'alpha_0_2': np.array(alpha_0_2)
             }
    k = np.arange(len(D))
    spec_flagfilter = np.array(spec_flagfilter)
    #spec_intfilter = np.array(list(map(lambda x: 1 if x else 0, spec_flagfilter)))
    
    # array of filters , one per file
    if remove_withfilters:     
        filter_indices = spec_flagfilter 
    else:
        filter_indices = np.ones_like(k, dtype=bool)    
    for par in params.keys():
        if par in ['amplitude']: #, 'alpha_0_2']:
            continue
        filter_indices *= np.logical_and(params[par] > np.median(params[par]) - sigma_clip * median_abs_deviation(params[par]),
                                         params[par] < np.median(params[par]) + sigma_clip * median_abs_deviation(params[par]))
    if  plot_flag:
        for par in params.keys():
            fig = plt.figure(figsize=(8,4))
            plt.plot(k, params[par])
            plt.plot(k[filter_indices], params[par][filter_indices], "ko")
            plt.grid()
            plt.title(par)

            suptitle = f"Observations : {dateobs} \n collection = {collection}"
            plt.suptitle(suptitle,fontsize=10,y=1.00)
            plt.tight_layout()
            plt.show()
            
    return [s for i,s in enumerate(specs) if filter_indices[i]]


def plot_spectra(spectra, colorparams,collection,dateobs,figsize=(10,6)):
    """
    plot spectra
    """
    
    #colormap = cm.Reds
    colormap = cm.jet 

    normalize = mcolors.Normalize(vmin=np.min(colorparams), vmax=np.max(colorparams))

    all_target_names = [] 

    fig  = plt.figure(figsize=figsize)
    count = 0
    for spec in spectra:
        target_name = spec.target.label
        if target_name in all_target_names:
            plt.plot(spec.lambdas, spec.data, color = colormap(normalize(spec.airmass)))
        else:
            plt.plot(spec.lambdas, spec.data, color = colormap(normalize(spec.airmass)),label=target_name)
            all_target_names.append(target_name)
        count +=1
            
    plt.grid()
    plt.xlabel("$\lambda$ [nm]")
    plt.ylabel(f"Flux [{spec.units}]")
    plt.legend()

    ax = plt.gca()
    
    # Colorbar setup
    s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
    s_map.set_array(colorparams)

    # If color parameters is a linspace, we can set boundaries in this way
    halfdist = (colorparams[1] - colorparams[0])/2.0
    boundaries = np.linspace(colorparams[0] - halfdist, colorparams[-1] + halfdist, len(colorparams) + 1)

    # Use this to emphasize the discrete color values
    cbar = fig.colorbar(s_map,ax=ax) #, spacing='proportional', ticks=colorparams, boundaries=boundaries, format='%2.2g') # format='%2i' for integer

    # Use this to show a continuous colorbar
    #cbar = fig.colorbar(s_map, spacing='proportional', ticks=colorparams, format='%2i')
    cbar.set_label("Airmass $z$")
    title = f"Observations : {dateobs}, nspec = {count}"
    suptitle = f"collection = {collection}"
    plt.title(title)
    plt.suptitle(suptitle,fontsize=10)
    #plt.tight_layout()
    plt.show()
    return fig

def plot_spectra_ax(spectra, ax,colorparams,dateobs):
    """
    plot spectra
    """
    
    #colormap = cm.Reds
    colormap = cm.jet 

    normalize = mcolors.Normalize(vmin=np.min(colorparams), vmax=np.max(colorparams))

    all_target_names = [] 

    count = 0
    for spec in spectra:
        target_name = spec.target.label
        if target_name in all_target_names:
            ax.plot(spec.lambdas, spec.data, color = colormap(normalize(spec.airmass)))
        else:
            ax.plot(spec.lambdas, spec.data, color = colormap(normalize(spec.airmass)),label=target_name)
            all_target_names.append(target_name)
        count +=1
            
    ax.grid()
    ax.set_xlabel("$\lambda$ [nm]")
    ax.set_ylabel(f"Flux [{spec.units}]")
    ax.legend()

    #ax = plt.gca()
    
    # Colorbar setup
    s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
    s_map.set_array(colorparams)

    # If color parameters is a linspace, we can set boundaries in this way
    halfdist = (colorparams[1] - colorparams[0])/2.0
    boundaries = np.linspace(colorparams[0] - halfdist, colorparams[-1] + halfdist, len(colorparams) + 1)

    # Use this to emphasize the discrete color values
    fig = ax.figure
    cbar = fig.colorbar(s_map,ax=ax) #, spacing='proportional', ticks=colorparams, boundaries=boundaries, format='%2.2g') # format='%2i' for integer
    #cbar = ax.collections[-1].colorbar 

    # Use this to show a continuous colorbar
    #cbar = fig.colorbar(s_map, spacing='proportional', ticks=colorparams, format='%2i')
    cbar.set_label("Airmass $z$")
    title = f"Observations : {dateobs}, nspec = {count}"
    #ax.set_title(title)
    #plt.tight_layout()



def plot_atmtransmission(spectra, colorparams,all_calspecs_sm,tel,disp,collection,dateobs,figsize=(10,6)):
    """
    plot spectra
    """

    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    #colormap = cm.Reds
    colormap = cm.jet 
   
    normalize = mcolors.Normalize(vmin=np.min(colorparams), vmax=np.max(colorparams))

    all_shown_target_names = [] 
    
    fig  = plt.figure(figsize=figsize)
    count = 0

    # loop on spectra
    for spec in spectra:

        #decode target name
        target_name = spec.target.label

        wls = spec.lambdas
        flx = spec.data
        flx_err = spec.err
        
        #c_dict = all_calspecs[target_name]
        c_dict = all_calspecs_sm[target_name]

        #smooth_data_np_convolve(sed,span)
        
        sed=np.interp(wls, c_dict["WAVELENGTH"]/10.,c_dict["FLUX"]*10.,left=1e-15,right=1e-15)
       
                     
        ratio = flx/tel.transmission(wls)/disp.transmission(wls)/sed
       
        indexes = np.where(np.logical_and(wls>350.,wls<=1000.))[0]
       
        sel_wls = wls[indexes]
        sel_ratio = ratio[indexes]
        
        if target_name in all_shown_target_names:
            plt.plot(sel_wls, sel_ratio, color = colormap(normalize(spec.airmass)))
        else:
            plt.plot(sel_wls,sel_ratio, color = colormap(normalize(spec.airmass)),label=target_name)
            all_shown_target_names.append(target_name)
        count +=1
            
     
            
    plt.grid()
    plt.xlabel("$\lambda$ [nm]")
    #plt.ylabel(f"Flux [{spec.units}]")
    plt.ylabel("atm. transmission")
    plt.legend()
    plt.xlim(360.,1000.)  
    plt.ylim(0.,1.2)  
    
    # Colorbar setup
    s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
    s_map.set_array(colorparams)

    # If color parameters is a linspace, we can set boundaries in this way
    halfdist = (colorparams[1] - colorparams[0])/2.0
    boundaries = np.linspace(colorparams[0] - halfdist, colorparams[-1] + halfdist, len(colorparams) + 1)

    ax = plt.gca()
    
    # Use this to emphasize the discrete color values
    cbar = fig.colorbar(s_map,ax=ax) #, spacing='proportional', ticks=colorparams, boundaries=boundaries, format='%2.2g') # format='%2i' for integer
    
    #cbar = fig.colorbar(s_map,spacing='proportional', ticks=colorparams, boundaries=boundaries, format='%2.2g') # format='%2i' for integer
    cbar.set_label("Airmass $z$")
    
    # Use this to show a continuous colorbar
    #cbar = fig.colorbar(s_map, spacing='proportional', ticks=colorparams, format='%2i')
    #cbar.set_label("Airmass $z$")
    title = f"Atmospheric transmission at target airmasses"
    suptitle = f"obs : {dateobs} , nspec = {count} \n coll = {collection}"
    plt.title(title)
    plt.suptitle(suptitle,fontsize=10,y=1.0)
    #plt.tight_layout()
    plt.show()


def plot_atmtransmission_ax(spectra, ax,colorparams,all_calspecs_sm,tel,disp,dateobs):
    """
    plot spectra
    """

    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    #colormap = cm.Reds
    colormap = cm.jet 
   
    normalize = mcolors.Normalize(vmin=np.min(colorparams), vmax=np.max(colorparams))

    all_shown_target_names = [] 
    count = 0

    # loop on spectra
    for spec in spectra:

        #decode target name
        target_name = spec.target.label

        wls = spec.lambdas
        flx = spec.data
        flx_err = spec.err
        
        #c_dict = all_calspecs[target_name]
        c_dict = all_calspecs_sm[target_name]

        #smooth_data_np_convolve(sed,span)
        
        sed=np.interp(wls, c_dict["WAVELENGTH"]/10.,c_dict["FLUX"]*10.,left=1e-15,right=1e-15)
       
                     
        ratio = flx/tel.transmission(wls)/disp.transmission(wls)/sed
       
        indexes = np.where(np.logical_and(wls>350.,wls<=1000.))[0]
       
        sel_wls = wls[indexes]
        sel_ratio = ratio[indexes]
        
        if target_name in all_shown_target_names:
            ax.plot(sel_wls, sel_ratio, color = colormap(normalize(spec.airmass)))
        else:
            ax.plot(sel_wls,sel_ratio, color = colormap(normalize(spec.airmass)),label=target_name)
            all_shown_target_names.append(target_name)
        count +=1
            
     
            
    ax.grid()
    ax.set_xlabel("$\lambda$ [nm]")
    #plt.ylabel(f"Flux [{spec.units}]")
    ax.set_ylabel("atm. transmission at z")
    ax.legend()
    ax.set_xlim(360.,1000.)  
    ax.set_ylim(0.,1.2)  
    
    # Colorbar setup
    s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
    s_map.set_array(colorparams)

    # If color parameters is a linspace, we can set boundaries in this way
    halfdist = (colorparams[1] - colorparams[0])/2.0
    boundaries = np.linspace(colorparams[0] - halfdist, colorparams[-1] + halfdist, len(colorparams) + 1)

    fig = ax.figure
    
    # Use this to emphasize the discrete color values
    cbar = fig.colorbar(s_map,ax=ax) #, spacing='proportional', ticks=colorparams, boundaries=boundaries, format='%2.2g') # format='%2i' for integer
    
    #cbar = fig.colorbar(s_map,spacing='proportional', ticks=colorparams, boundaries=boundaries, format='%2.2g') # format='%2i' for integer
    cbar.set_label("Airmass $z$")
    
    # Use this to show a continuous colorbar
    #cbar = fig.colorbar(s_map, spacing='proportional', ticks=colorparams, format='%2i')
    #cbar.set_label("Airmass $z$")
    title = f"Atmospheric transmission at target airmasses, obs : {dateobs} , nspec = {count}"
    #ax.set_title(title)
    
    

def plot_atmtransmission_zcorr(spectra, colorparams,all_calspecs_sm,tel,disp,collection,dateobs,figsize=(10,6)):
    """
    plot atmospheric transmission corrected for airmass = 1 (z_pred = 1)

    $$
    T(z_{pred}) = \frac{ \left( T(z_{meas}) \right)^\left( \frac{z_{pred}}{z_{meas}}\right)}{(T^{grey}_{z_{meas}})^{z_{pred}}}
    $$
    parameters
     - spectra,
     - colorparmas
    
    """

    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    #colormap = cm.Reds
    colormap = cm.jet 
   

    normalize = mcolors.Normalize(vmin=np.min(colorparams), vmax=np.max(colorparams))
    all_shown_target_names = [] 
    fig  = plt.figure(figsize=figsize)

    count = 0
    for spec in spectra:
             
        target_name = spec.target.label

        wls = spec.lambdas
        flx = spec.data
        flx_err = spec.err
        
        #c_dict = all_calspecs[target_name]
        c_dict = all_calspecs_sm[target_name]

        #smooth_data_np_convolve(sed,span)
        
        sed=np.interp(wls, c_dict["WAVELENGTH"]/10.,c_dict["FLUX"]*10.,left=1e-15,right=1e-15)
                         
        ratio = flx/tel.transmission(wls)/disp.transmission(wls)/sed
       
        indexes = np.where(np.logical_and(wls>350.,wls<=1000.))[0]
       
        sel_wls = wls[indexes]
        sel_ratio = ratio[indexes]
        sel_ratio_airmas_corr = np.power(sel_ratio,1/spec.airmass)
        
        if target_name in all_shown_target_names:
            plt.plot(sel_wls, sel_ratio_airmas_corr, color = colormap(normalize(spec.airmass)))
        else:
            plt.plot(sel_wls,sel_ratio_airmas_corr, color = colormap(normalize(spec.airmass)),label=target_name)
            all_shown_target_names.append(target_name)
        count += 1
            
     
            
    plt.grid()
    plt.xlabel("$\lambda$ [nm]")
    plt.ylabel("atm. transmission")
    plt.legend()
    plt.xlim(360.,1000.)  
    plt.ylim(0.,1.2)  

    ax = plt.gca()
    
    # Colorbar setup
    s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
    s_map.set_array(colorparams)

    # If color parameters is a linspace, we can set boundaries in this way
    halfdist = (colorparams[1] - colorparams[0])/2.0
    boundaries = np.linspace(colorparams[0] - halfdist, colorparams[-1] + halfdist, len(colorparams) + 1)

    # Use this to emphasize the discrete color values
    cbar = fig.colorbar(s_map,ax=ax) #, spacing='proportional', ticks=colorparams, boundaries=boundaries, format='%2.2g') # format='%2i' for integer

    # Use this to show a continuous colorbar
    #cbar = fig.colorbar(s_map, spacing='proportional', ticks=colorparams, format='%2i')
    cbar.set_label("Airmass $z$")
    title = f"Atmospheric transmission scaled for airmass=1"
    suptitle = f"obs : {dateobs} , nspec = {count} \n coll = {collection}"
    plt.title(title)
    plt.suptitle(suptitle,fontsize=10,y=1.0)
    #plt.tight_layout()
    plt.show()
    

def plot_atmtransmission_zcorr_ax(spectra, ax, colorparams,all_calspecs_sm,tel,disp,dateobs):
    """
    plot atmospheric transmission corrected for airmass = 1 (z_pred = 1)

    $$
    T(z_{pred}) = \frac{ \left( T(z_{meas}) \right)^\left( \frac{z_{pred}}{z_{meas}}\right)}{(T^{grey}_{z_{meas}})^{z_{pred}}}
    $$
    parameters
     - spectra,
     - colorparmas
    
    """

    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    #colormap = cm.Reds
    colormap = cm.jet 
   

    normalize = mcolors.Normalize(vmin=np.min(colorparams), vmax=np.max(colorparams))
    all_shown_target_names = [] 
  

    count = 0
    for spec in spectra:
             
        target_name = spec.target.label

        wls = spec.lambdas
        flx = spec.data
        flx_err = spec.err
        
        #c_dict = all_calspecs[target_name]
        c_dict = all_calspecs_sm[target_name]

        #smooth_data_np_convolve(sed,span)
        
        sed=np.interp(wls, c_dict["WAVELENGTH"]/10.,c_dict["FLUX"]*10.,left=1e-15,right=1e-15)
                         
        ratio = flx/tel.transmission(wls)/disp.transmission(wls)/sed
       
        indexes = np.where(np.logical_and(wls>350.,wls<=1000.))[0]
       
        sel_wls = wls[indexes]
        sel_ratio = ratio[indexes]
        sel_ratio_airmas_corr = np.power(sel_ratio,1/spec.airmass)
        
        if target_name in all_shown_target_names:
            ax.plot(sel_wls, sel_ratio_airmas_corr, color = colormap(normalize(spec.airmass)))
        else:
            ax.plot(sel_wls,sel_ratio_airmas_corr, color = colormap(normalize(spec.airmass)),label=target_name)
            all_shown_target_names.append(target_name)
        count += 1
            
     
            
    ax.grid()
    ax.set_xlabel("$\lambda$ [nm]")
    ax.set_ylabel("atm. transmission at z=1.0")
    ax.legend()
    ax.set_xlim(360.,1000.)  
    ax.set_ylim(0.,1.2)  

    fig = ax.figure
    
    # Colorbar setup
    s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
    s_map.set_array(colorparams)

    # If color parameters is a linspace, we can set boundaries in this way
    halfdist = (colorparams[1] - colorparams[0])/2.0
    boundaries = np.linspace(colorparams[0] - halfdist, colorparams[-1] + halfdist, len(colorparams) + 1)

    # Use this to emphasize the discrete color values
    cbar = fig.colorbar(s_map,ax=ax) #, spacing='proportional', ticks=colorparams, boundaries=boundaries, format='%2.2g') # format='%2i' for integer

    # Use this to show a continuous colorbar
    #cbar = fig.colorbar(s_map, spacing='proportional', ticks=colorparams, format='%2i')
    cbar.set_label("Airmass $z$")
    title = f"Atmospheric transmission scaled for airmass=1, obs : {dateobs} , nspec = {count}"
    #ax.set_title(title)
   



#def plot_atmtransmission_zcorr_antatmsim(spectra, colorparams,all_calspecs_sm,tel,disp,collection,dateobs,df_atm,am=1,pwv=2,oz=300,vaod=0.01,grey=0.99):
def plot_atmtransmission_zcorr_antatmsim(spectra, colorparams,all_calspecs_sm,tel,disp,collection,dateobs,df_atm,am=1):
    """
    plot spectra predicted at airmass = 1

    $$
    T(z_{pred}) = \frac{ \left( T(z_{meas}) \right)^\left( \frac{z_{pred}}{z_{meas}}\right)}{(T^{grey}_{z_{meas}})^{z_{pred}}}
    $$
    
    """

    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    colormap = cm.Reds
    #colormap = cm.jet 

    # find average atmospheric parameters
    df_good = df_atm[df_atm.filtered].drop(["id","filtered"],axis=1)
    m_A1 , m_ozone, m_PWV, m_VAOD = df_good.median().values
    print(" mean atm parameters",m_A1 , m_ozone, m_PWV, m_VAOD)

    all_meas_atmtransmissions = []

    normalize = mcolors.Normalize(vmin=np.min(colorparams), vmax=np.max(colorparams))

    all_shown_target_names = [] 
    
    fig  = plt.figure(figsize=(11,6))
    count = 0
    for spec in spectra:
        
        row = df_atm[df_atm.id == spec.dataId]
        (s_id, s_target, s_A1, s_ozone, s_PWV, s_VAOD, s_flag) = row.values[0]
        
        if s_flag:
            pwv=s_PWV
            oz=s_ozone
            vaod=s_VAOD
            grey=m_A1
        else:
            pwv=m_PWV
            oz=m_ozone
            vaod=m_VAOD
            grey=m_A1
            
        target_name = spec.target.label

        wls = spec.lambdas
        flx = spec.data
        flx_err = spec.err
        
        #c_dict = all_calspecs[target_name]
        c_dict = all_calspecs_sm[target_name]

        #smooth_data_np_convolve(sed,span)

        sed=np.interp(wls, c_dict["WAVELENGTH"]/10.,c_dict["FLUX"]*10.,left=1e-15,right=1e-15)                 
        ratio = flx/tel.transmission(wls)/disp.transmission(wls)/sed
        
        indexes = np.where(np.logical_and(wls>350.,wls<=1000.))[0]
       
        sel_wls = wls[indexes]
        sel_ratio = ratio[indexes]
        sel_ratio_airmas_corr = np.power(sel_ratio,am/spec.airmass)/(np.power(grey,am))
        
        if target_name in all_shown_target_names:
            if s_flag:
                plt.plot(sel_wls, sel_ratio_airmas_corr, color = colormap(normalize(spec.airmass)))
        else:
            if s_flag:
                plt.plot(sel_wls,sel_ratio_airmas_corr, color = colormap(normalize(spec.airmass)),label=target_name)
                all_shown_target_names.append(target_name)
        #anyway add in transmission even if flag is s_flag false
        all_meas_atmtransmissions.append((sel_wls,sel_ratio_airmas_corr))
        count +=1
    
    textstr = '\n'.join((
    r'$am=%.2f$' % (am, ),
    r'$grey=%.2f$' % (m_A1, ),
    r'$pwv=%.2f$ mm' % (m_PWV, ),
    r'$ozone=%.1f$ DU' % (m_ozone, ),
    r'$vaod=%.3f$' % (m_VAOD,)))
    emul1 =  ObsAtmo("AUXTEL",740.)
    emul2 =  ObsAtmo("AUXTEL",730.)
    transm_sim1 = emul1.GetAllTransparencies(sel_wls,am,m_PWV,m_ozone,tau=m_VAOD)
    transm_sim2 = emul2.GetAllTransparencies(sel_wls,am,m_PWV,m_ozone,tau=m_VAOD)
    
    #plt.plot(sel_wls,transm_sim1,'-g',label=f"simulation P=740. hPa")
    plt.plot(sel_wls,transm_sim2,'-b',label=f"simulation P=730. hPa")

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax = plt.gca()
    # place a text box in upper left in axes coords
    ax.text(0.70, 0.25, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
            
    plt.grid()
    plt.xlabel("$\lambda$ [nm]")
    #plt.ylabel(f"Flux [{spec.units}]")
    plt.legend()
    plt.xlim(360.,1000.)  
    plt.ylim(0.,1.3)  
    
    # Colorbar setup
    s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
    s_map.set_array(colorparams)

    # If color parameters is a linspace, we can set boundaries in this way
    halfdist = (colorparams[1] - colorparams[0])/2.0
    boundaries = np.linspace(colorparams[0] - halfdist, colorparams[-1] + halfdist, len(colorparams) + 1)

    # Use this to emphasize the discrete color values
    cbar = fig.colorbar(s_map,ax=ax) #, spacing='proportional', ticks=colorparams, boundaries=boundaries, format='%2.2g') # format='%2i' for integer

    # Use this to show a continuous colorbar
    #cbar = fig.colorbar(s_map, spacing='proportional', ticks=colorparams, format='%2i')
    cbar.set_label("Airmass $z$")
    title = f"Atmospheric transmission scaled for airmass={am})"
    suptitle = f"obs : {dateobs} , nspec = {count} \n coll = {collection}"
    plt.title(title)
    plt.suptitle(suptitle,fontsize=10,y=1.0)
    plt.show()
    return all_meas_atmtransmissions
    


#def plot_atmtransmission_zcorr_antatmsim_ratio(spectra,colorparams,all_calspecs_sm,tel,disp,collection,dateobs,df_atm,am=1,pwv=2,oz=300,vaod=0.01,grey=0.99):
def plot_atmtransmission_zcorr_antatmsim_ratio(spectra,colorparams,all_calspecs_sm,tel,disp,collection,dateobs,df_atm,am=1):
    """
    plot spectra

    $$
    T(z_{pred}) = \frac{ \left( T(z_{meas}) \right)^\left( \frac{z_{pred}}{z_{meas}}\right)}{(T^{grey}_{z_{meas}})^{z_{pred}}}
    $$
    
    """

    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    colormap = cm.Reds
    #colormap = cm.jet 

    # find average atmospheric parameters
    df_good = df_atm[df_atm.filtered].drop(["id","filtered"],axis=1)

    try:
        m_A1 , m_ozone, m_PWV, m_VAOD = df_good.median().values
        print(" mean parameters",m_A1 , m_ozone, m_PWV, m_VAOD)
    except Exception as inst:
        print(">>>>   !!!! Exception plot_atmtransmission_zcorr_antatmsim_ratio !!!!")
        print(type(inst))    # the exception type
        print(inst.args)     # arguments stored in .args
        print(inst)          # __str__ allows args to be printed directly,
                             # but may be overridden in exception subclasses
        m_A1 , m_ozone, m_PWV, m_VAOD = 1.,0.,0.,0.
    

    normalize = mcolors.Normalize(vmin=np.min(colorparams), vmax=np.max(colorparams))

    all_shown_target_names = [] 
    
    fig  = plt.figure(figsize=(14,10))

    grid = gridspec.GridSpec(2, 1, height_ratios=[2.5,1])

    ax1 = plt.subplot(grid[0])
    ax2 = plt.subplot(grid[1],sharex=ax1)

    textstr = '\n'.join((
    r'$am=%.2f$' % (am, ),
    r'$grey=%.2f$' % (m_A1, ),
    r'$pwv=%.2f$ mm' % (m_PWV, ),
    r'$ozone=%.1f$ DU' % (m_ozone, ),
    r'$vaod=%.3f$' % (m_VAOD,)))
    emul1 =  ObsAtmo("AUXTEL",740.)
    emul2 =  ObsAtmo("AUXTEL",730.)
    

    count = 0
    for spec in spectra:     

        row = df_atm[df_atm.id == spec.dataId]
        (s_id, s_target, s_A1, s_ozone, s_PWV, s_VAOD, s_flag) = row.values[0]
        if s_flag:
            pwv=s_PWV
            oz=s_ozone
            vaod=s_VAOD
            grey=m_A1
        else:
            pwv=m_PWV
            oz=m_ozone
            vaod=m_VAOD
            grey=m_A1
        
        
        target_name = spec.target.label

        wls = spec.lambdas
        flx = spec.data
        flx_err = spec.err
        
        #c_dict = all_calspecs[target_name]
        c_dict = all_calspecs_sm[target_name]

        #smooth_data_np_convolve(sed,span)
        
        sed=np.interp(wls, c_dict["WAVELENGTH"]/10.,c_dict["FLUX"]*10.,left=1e-15,right=1e-15)
                      
        ratio = flx/tel.transmission(wls)/disp.transmission(wls)/sed
       
        indexes = np.where(np.logical_and(wls>350.,wls<=1000.))[0]
       
        sel_wls = wls[indexes]
        sel_ratio = ratio[indexes]
        sel_ratio_airmas_corr = np.power(sel_ratio,am/spec.airmass)/(np.power(grey,am))
        
        if target_name in all_shown_target_names:
            if s_flag:
                ax1.plot(sel_wls, sel_ratio_airmas_corr, color = colormap(normalize(spec.airmass)))
        else:
            if s_flag:
                ax1.plot(sel_wls,sel_ratio_airmas_corr, color = colormap(normalize(spec.airmass)),label=target_name)
                all_shown_target_names.append(target_name)

        transm_sim1 = emul1.GetAllTransparencies(sel_wls,am,pwv,oz,tau=vaod)
        transm_sim2 = emul2.GetAllTransparencies(sel_wls,am,pwv,oz,tau=vaod)
    
        #ax1.plot(sel_wls,transm_sim1,'-g')
        #ax1.plot(sel_wls,transm_sim2,'-b')

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
   
        # place a text box in upper left in axes coords
        if s_flag:
            ax2.plot(sel_wls,sel_ratio_airmas_corr/transm_sim2,color = colormap(normalize(spec.airmass)),label=f"simulation P=740. hPa")
        #ax2.plot(sel_wls,sel_ratio_airmas_corr/transm_sim2,'-g',label=f"simulation P=730. hPa")
     

        count += 1
            
    ax1.grid()
    # show text box in upper plot
    ax1.text(0.70, 0.25, textstr, transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    # show simulation in upper plot
    transm_sim1 = emul1.GetAllTransparencies(sel_wls,am,m_PWV,m_ozone,tau=m_VAOD)
    transm_sim2 = emul2.GetAllTransparencies(sel_wls,am,m_PWV,m_ozone,tau=m_VAOD)
    ax1.plot(sel_wls,transm_sim2,'-b')
    
    #ax1.set_xlabel("$\lambda$ [nm]")
    #plt.ylabel(f"Flux [{spec.units}]")
    ax1.legend()
    ax1.set_xlim(360.,1000.)  
    ax1.set_ylim(0.,1.3)  

    ax2.set_title(f"ratio spectrum/sim at airmass {am:.2f}")
    ax2.set_xlabel("$\lambda$ [nm]")  
    ax2.set_ylim(0.9,1.1)  
    ax2.grid()
    # Colorbar setup
    s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
    s_map.set_array(colorparams)

    # If color parameters is a linspace, we can set boundaries in this way
    halfdist = (colorparams[1] - colorparams[0])/2.0
    boundaries = np.linspace(colorparams[0] - halfdist, colorparams[-1] + halfdist, len(colorparams) + 1)

    # Use this to emphasize the discrete color values
    #cbar = fig.colorbar(s_map) #, spacing='proportional', ticks=colorparams, boundaries=boundaries, format='%2.2g') # format='%2i' for integer

    # Use this to show a continuous colorbar
    #cbar = fig.colorbar(s_map, spacing='proportional', ticks=colorparams, format='%2i')
    #cbar.set_label("Airmass $z$")
    title = f"Atmospheric transmission scaled for airmass={am})"
    suptitle = f"obs : {dateobs} , nspec= {count} \n coll = {collection}"
    ax1.set_title(title)
    plt.suptitle(suptitle,fontsize=10,y=1.0)

    plt.show()


def savehdf5_pernightspectra(spectra,df_spec_night,all_calspecs_sm,tel,disp,dateobs,pathdata):
    """
    Save Spectra, atmospheric transmission in hdf5 files 

    refer to 
    $$
    T(z_{pred}) = \frac{ \left( T(z_{meas}) \right)^\left( \frac{z_{pred}}{z_{meas}}\right)}{(T^{grey}_{z_{meas}})^{z_{pred}}}
    $$
    
    """

    # create the file
    file_hdf5 = f"spectra_transmission_{dateobs}.h5"
    ffile_hdf5 = os.path.join(pathdata,file_hdf5)

    print(f">>>> create file hdf5 {ffile_hdf5}")
    hf = h5py.File(ffile_hdf5, 'w')

    # Find the relattive time wrt midnight
    tmin = df_spec_night["Time"].min()
    tmax = df_spec_night["Time"].max()
    df_spec_night.assign(dt = lambda row : (row["Time"]-tmin).dt.seconds/3600.,inplace=True)

    list_of_targets = df_spec_night["TARGET"].unique()

    list_visitid = list(df_spec_night["id"])
     
    # convert in hours wrt midnight
  

    for idx,visitid in enumerate(list_visitid):
        group_name = f'spectrum_{visitid}'
        spec = spectra[idx]
        target_name = spec.target.label
        airmass = spec.airmass 
   
        #print(f">>>> create group {group_name}")
        g_spec = hf.create_group(group_name)

        g_spec.attrs['airmass'] = airmass
        g_spec.attrs['visitid'] = visitid
        g_spec.attrs["target"] = target_name

        
        # extract the flux
        wls = spec.lambdas
        flx = spec.data
        flx_err = spec.err

        # save the flux
        d = g_spec.create_dataset("wls",data=wls,compression="gzip", compression_opts=9)
        d = g_spec.create_dataset("fls",data=flx,compression="gzip", compression_opts=9)
        d = g_spec.create_dataset("fls_err",data=flx_err,compression="gzip", compression_opts=9)

        # extract SED
        c_dict = all_calspecs_sm[target_name] 
        sed=np.interp(wls, c_dict["WAVELENGTH"]/10.,c_dict["FLUX"]*10.,left=1e-15,right=1e-15)
                         
        ratio_atz = flx/tel.transmission(wls)/disp.transmission(wls)/sed
        ratio_atz_err = flx_err/tel.transmission(wls)/disp.transmission(wls)/sed

        #save datasets in hdf5
        d = g_spec.create_dataset("transm_atz",data=ratio_atz,compression="gzip", compression_opts=9)
        d = g_spec.create_dataset("transm_atz_err",data=ratio_atz_err,compression="gzip", compression_opts=9)

        ratio_atz1 = np.power(ratio_atz,1./airmass)
        ratio_atz1_err = 1/airmass * ratio_atz_err/np.power(ratio_atz,1.-1./airmass)
    
        d = g_spec.create_dataset("transm_atz1",data=ratio_atz1,compression="gzip", compression_opts=9)
        d = g_spec.create_dataset("transm_atz1_err",data=ratio_atz1_err,compression="gzip", compression_opts=9)

        d = g_spec.create_dataset("sed",data=sed,compression="gzip", compression_opts=9)
        d = g_spec.create_dataset("disptransm",data=disp.transmission(wls),compression="gzip", compression_opts=9)
        d = g_spec.create_dataset("teltransm",data=tel.transmission(wls),compression="gzip", compression_opts=9)

    
    #print(f">>>> save file hdf5 {ffile_hdf5}")
    hf.close() 
    return ffile_hdf5


def savehdf5_pernightspectra_withsim(spectra,df_spec_night,all_calspecs_sm,tel,disp,dateobs,emul,pathdata):
    """
    Save Spectra, atmospheric transmission in hdf5 files 

    refer to 
    $$
    T(z_{pred}) = \frac{ \left( T(z_{meas}) \right)^\left( \frac{z_{pred}}{z_{meas}}\right)}{(T^{grey}_{z_{meas}})^{z_{pred}}}
    $$

    Use also getObsAtmo to transport transmission from z to 1
    """

    # create the file
    file_hdf5 = f"spectra_transmission_{dateobs}.h5"
    ffile_hdf5 = os.path.join(pathdata,file_hdf5)

    print(f">>>> create file hdf5 {ffile_hdf5}")
    hf = h5py.File(ffile_hdf5, 'w')

    # Find the relattive time wrt midnight
    tmin = df_spec_night["Time"].min()
    tmax = df_spec_night["Time"].max()
    df_spec_night.assign(dt = lambda row : (row["Time"]-tmin).dt.seconds/3600.,inplace=True)

    list_of_targets = df_spec_night["TARGET"].unique()

    list_visitid = list(df_spec_night["id"])

    #preselect 
    df_spec_night_fittedparams = df_spec_night[['id',"dt_midnight",
                                              'A1_x', 'A1_err_x','A2_x','A2_err_x','A3','A3_err',
                                              'VAOD_x','VAOD_err_x',
                                              'angstrom_exp_x','angstrom_exp_err_x',
                                              'ozone [db]_x','ozone [db]_err_x',
                                              'PWV [mm]_x','PWV [mm]_err_x']]

    
     
    # convert in hours wrt midnight
  

    for idx,visitid in enumerate(list_visitid):
        group_name = f'spectrum_{visitid}'
        spec = spectra[idx]
        target_name = spec.target.label
        airmass = spec.airmass 
   
        #print(f">>>> create group {group_name}")
        g_spec = hf.create_group(group_name)
        g_spec.attrs['airmass'] = airmass
        g_spec.attrs['visitid'] = visitid
        g_spec.attrs["target"] = target_name

        df_params = df_spec_night_fittedparams[df_spec_night_fittedparams["id"] == visitid ]
       
        list_of_params = list(df_params.columns)

        for param_name in list_of_params:
            value = df_params.iloc[0][param_name]
            g_spec.attrs[param_name] = value
         

        pwv = df_params.iloc[0]['PWV [mm]_x']
        oz =  df_params.iloc[0]['ozone [db]_x']
        tau = df_params.iloc[0]['VAOD_x']
        beta= df_params.iloc[0]['angstrom_exp_x']
        
             
        # extract the flux
        wls = spec.lambdas
        flx = spec.data
        flx_err = spec.err

        # save the flux
        d = g_spec.create_dataset("wls",data=wls,compression="gzip", compression_opts=9)
        d = g_spec.create_dataset("fls",data=flx,compression="gzip", compression_opts=9)
        d = g_spec.create_dataset("fls_err",data=flx_err,compression="gzip", compression_opts=9)

        # extract SED
        c_dict = all_calspecs_sm[target_name] 
        sed=np.interp(wls, c_dict["WAVELENGTH"]/10.,c_dict["FLUX"]*10.,left=1e-15,right=1e-15)
                         
        ratio_atz = flx/tel.transmission(wls)/disp.transmission(wls)/sed
        ratio_atz_err = flx_err/tel.transmission(wls)/disp.transmission(wls)/sed

        d = g_spec.create_dataset("transm_atz",data=ratio_atz,compression="gzip", compression_opts=9)
        d = g_spec.create_dataset("transm_atz_err",data=ratio_atz_err,compression="gzip", compression_opts=9)


        # cut low wavelengths
        wls_int  = np.where(wls<300.,300.,wls)
        TZ = emul.GetAllTransparencies(wls_int,airmass,pwv,oz,tau,beta)
        T1 = emul.GetAllTransparencies(wls_int,1.,pwv,oz,tau,beta)

        

        ratio_tsim =  TZ/T1

        #simple correction
        ratio_atz1 = np.power(ratio_atz,1./airmass)
        ratio_atz1_err = 1/airmass * ratio_atz_err/np.power(ratio_atz,1.-1./airmass)

        #advanced correction
        ratio_atz12 = ratio_atz/ratio_tsim
        ratio_atz12_err = ratio_atz_err/ratio_tsim 

         
        d = g_spec.create_dataset("transm_atz1",data=ratio_atz1,compression="gzip", compression_opts=9)
        d = g_spec.create_dataset("transm_atz1_err",data=ratio_atz1_err,compression="gzip", compression_opts=9)

        d = g_spec.create_dataset("transm_atz12",data=ratio_atz12,compression="gzip", compression_opts=9)
        d = g_spec.create_dataset("transm_atz12_err",data=ratio_atz12_err,compression="gzip", compression_opts=9)

        d = g_spec.create_dataset("transmsim_ratio",data=ratio_tsim ,compression="gzip", compression_opts=9)

        d = g_spec.create_dataset("sed",data=sed,compression="gzip", compression_opts=9)
        d = g_spec.create_dataset("disptransm",data=disp.transmission(wls),compression="gzip", compression_opts=9)
        d = g_spec.create_dataset("teltransm",data=tel.transmission(wls),compression="gzip", compression_opts=9)
        
    
    #print(f">>>> save file hdf5 {ffile_hdf5}")
    hf.close() 
    return ffile_hdf5
    

def readhdf5_pernightspectra(the_h5_file):
    """
    read Spectra, atmospheric transmission from hdf5 files 

    Parameters:
        the_h5_file : full filename (path and filename)
    """

    hf = h5py.File(the_h5_file, 'r')
    list_of_keys = list(hf.keys())
    
    dict_spectra = {}
    for a_group_key, h5obj in hf.items():
        if isinstance(h5obj,h5py.Group):
            #print(a_group_key,'is a Group')
            group_number = int(a_group_key.split('_')[1])
            
            dict_spectrum_attributes = {}
            dict_spectrum_datasets = {}
            group = hf[a_group_key]
        
            list_of_datasets = group.keys()
            list_of_attributes = group.attrs.keys()
        
            #print("\t attributes : ",list_of_attributes)
            for key in list_of_attributes:
                dict_spectrum_attributes[key] = group.attrs[key]
            #print("\t datasets   : ", list_of_datasets)
            for key in list_of_datasets:
                dict_spectrum_datasets[key] = group[key][:]
            
            dict_spectra[group_number] = dict(attr=dict_spectrum_attributes, datasets=dict_spectrum_datasets)   
            
        elif isinstance(h5obj,h5py.Dataset):
            #print(a_group_key,'is a Dataset')
            pass
    return  dict_spectra


def savehdf5_pernightspectraold(spectra,df_spec_night,all_calspecs_sm,tel,disp,dateobs,pathdata):
    """
    Save Spectra, atmospheric transmission in hdf5 files 

    refer to 
    $$
    T(z_{pred}) = \frac{ \left( T(z_{meas}) \right)^\left( \frac{z_{pred}}{z_{meas}}\right)}{(T^{grey}_{z_{meas}})^{z_{pred}}}
    $$
    
    """

    # create the file
    file_hdf5 = f"spectra_transmission_{dateobs}.h5"
    ffile_hdf5 = os.path.join(pathdata,file_hdf5)

    print(f">>>> create file hdf5 {ffile_hdf5}")
    hf = h5py.File(file_hdf5, 'w')

    # Find the relattive time wrt midnight
    tmin = df_spec_night["Time"].min()
    tmax = df_spec_night["Time"].max()
    list_of_targets = df_spec_night["TARGET"].unique()
    str_list_of_targets = "\n".join(list_of_targets)
     
    # convert in hours wrt midnight
    df_spec_night.assign(dt = lambda row : (row["Time"]-tmin).dt.seconds/3600.,inplace=True)


    for spec in spectra:
        target_name = spec.target.label
        airmass = spec.airmass 
    # first froup : the median parameters
    group_name = 'spectra'
    print(f">>>> create group {group_name}")
    g_atmparam = hf.create_group(group_name)

    # find average atmospheric parameters
    df_good = df_atm[df_atm.filtered].drop(["id","filtered"],axis=1)

    try:
        m_A1 , m_ozone, m_PWV, m_VAOD = df_good.median().values
        print(" mean parameters",m_A1 , m_ozone, m_PWV, m_VAOD)
    except Exception as inst:
        print(">>>>   !!!! savehdf5_atmtransmission_zcorr_antatmsim_ratio !!!!")
        print(type(inst))    # the exception type
        print(inst.args)     # arguments stored in .args
        print(inst)          # __str__ allows args to be printed directly,
                             # but may be overridden in exception subclasses
        m_A1 , m_ozone, m_PWV, m_VAOD = 1.,0.,0.,0.
    
   
   
    g_atmparam.attrs['md_grey'] = m_A1
    g_atmparam.attrs['md_ozone'] = m_ozone
    g_atmparam.attrs["md_PWV"] = m_PWV
    g_atmparam.attrs["md_VAOD"] = m_VAOD

    # loop on Spec
    count = 0
    for spec in spectra:     

        row = df_atm[df_atm.id == spec.dataId]
        (s_id, s_target, s_A1, s_ozone, s_PWV, s_VAOD, s_flag) = row.values[0]
        if s_flag:
            pwv=s_PWV
            oz=s_ozone
            vaod=s_VAOD
            grey=m_A1
        else:
            pwv=m_PWV
            oz=m_ozone
            vaod=m_VAOD
            grey=m_A1

        target_name = spec.target.label

        group_name = f"spec_{spec.dataId}"
        print(f">>>> create group {group_name}")
        
        spec_group =  hf.create_group(f"spec_{spec.dataId}")
        spec_group.attrs['target'] = target_name
        spec_group.attrs['airmass'] = spec.airmass
        
        spec_group.attrs['flag_atmparam'] = int(s_flag)
        spec_group.attrs['grey'] = grey
        spec_group.attrs['ozone'] = oz
        spec_group.attrs['VAOD'] = vaod
        spec_group.attrs['PWV'] = pwv
        
        
        wls = spec.lambdas
        flx = spec.data
        flx_err = spec.err
        
        d = spec_group.create_dataset("wls",data=wls,compression="gzip", compression_opts=9)
        d = spec_group.create_dataset("fls",data=flx,compression="gzip", compression_opts=9)
        d = spec_group.create_dataset("fls_err",data=flx_err,compression="gzip", compression_opts=9)
        
        c_dict = all_calspecs_sm[target_name]

        #smooth_data_np_convolve(sed,span)
        sed=np.interp(wls, c_dict["WAVELENGTH"]/10.,c_dict["FLUX"]*10.,left=1e-15,right=1e-15)
                      
        ratio = flx/tel.transmission(wls)/disp.transmission(wls)/sed
        ratio_err = flx/tel.transmission(wls)/disp.transmission(wls)/sed
       
        indexes = np.where(np.logical_and(wls>350.,wls<=1000.))[0]
       
        sel_wls = wls[indexes]
        sel_ratio = ratio[indexes]
        sel_ratio_airmas_corr = np.power(sel_ratio,am/spec.airmass)/(np.power(grey,am))
        sel_ratio_err = ratio_err[indexes]

        d = spec_group.create_dataset("wlr",data=sel_wls,compression="gzip", compression_opts=9)
        d = spec_group.create_dataset("ratio",data=sel_ratio,compression="gzip", compression_opts=9)
        d = spec_group.create_dataset("ratio_airmass1",data=sel_ratio_airmas_corr,compression="gzip", compression_opts=9)
        d = spec_group.create_dataset("ratio_err",data=sel_ratio_err,compression="gzip", compression_opts=9)
        count += 1
    group_name = "sim_spec"
    print(f">>>> create group {group_name}")
    sim_group =  hf.create_group(group_name)
   
    transm_sim1 = emul1.GetAllTransparencies(sel_wls,am,m_PWV,m_ozone,tau=m_VAOD)
    transm_sim2 = emul2.GetAllTransparencies(sel_wls,am,m_PWV,m_ozone,tau=m_VAOD)

    sim_group.attrs['ozone'] = m_ozone
    sim_group.attrs['VAOD'] = m_VAOD
    sim_group.attrs['PWV'] = m_PWV
    d = sim_group.create_dataset("wls",data=sel_wls,compression="gzip", compression_opts=9)
    d = sim_group.create_dataset("transm",data=transm_sim2,compression="gzip", compression_opts=9)   
    hf.close() 

def savehdf5_atmtransmission_zcorr_antatmsim_ratio(spectra,colorparams,all_calspecs_sm,tel,disp,collection,dateobs,df_atm,am=1):
    """
    Save Spectra, atmospheric transmission in hdf5 files 

    refer to 
    $$
    T(z_{pred}) = \frac{ \left( T(z_{meas}) \right)^\left( \frac{z_{pred}}{z_{meas}}\right)}{(T^{grey}_{z_{meas}})^{z_{pred}}}
    $$
    
    """

    emul1 =  ObsAtmo("AUXTEL",740.)
    emul2 =  ObsAtmo("AUXTEL",730.)

    file_hdf5 = f"spectra_transmission_ratio_{dateobs}.h5"

    print(f">>>> create file {file_hdf5}")
    hf = h5py.File(file_hdf5, 'w')

    # first froup : the median parameters
    group_name = 'median_param_atm'
    print(f">>>> create group {group_name}")
    g_atmparam = hf.create_group(group_name)

    # find average atmospheric parameters
    df_good = df_atm[df_atm.filtered].drop(["id","filtered"],axis=1)

    try:
        m_A1 , m_ozone, m_PWV, m_VAOD = df_good.median().values
        print(" mean parameters",m_A1 , m_ozone, m_PWV, m_VAOD)
    except Exception as inst:
        print(">>>>   !!!! savehdf5_atmtransmission_zcorr_antatmsim_ratio !!!!")
        print(type(inst))    # the exception type
        print(inst.args)     # arguments stored in .args
        print(inst)          # __str__ allows args to be printed directly,
                             # but may be overridden in exception subclasses
        m_A1 , m_ozone, m_PWV, m_VAOD = 1.,0.,0.,0.
    
   
   
    g_atmparam.attrs['md_grey'] = m_A1
    g_atmparam.attrs['md_ozone'] = m_ozone
    g_atmparam.attrs["md_PWV"] = m_PWV
    g_atmparam.attrs["md_VAOD"] = m_VAOD

    # loop on Spec
    count = 0
    for spec in spectra:     

        row = df_atm[df_atm.id == spec.dataId]
        (s_id, s_target, s_A1, s_ozone, s_PWV, s_VAOD, s_flag) = row.values[0]
        if s_flag:
            pwv=s_PWV
            oz=s_ozone
            vaod=s_VAOD
            grey=m_A1
        else:
            pwv=m_PWV
            oz=m_ozone
            vaod=m_VAOD
            grey=m_A1

        target_name = spec.target.label

        group_name = f"spec_{spec.dataId}"
        print(f">>>> create group {group_name}")
        
        spec_group =  hf.create_group(f"spec_{spec.dataId}")
        spec_group.attrs['target'] = target_name
        spec_group.attrs['airmass'] = spec.airmass
        
        spec_group.attrs['flag_atmparam'] = int(s_flag)
        spec_group.attrs['grey'] = grey
        spec_group.attrs['ozone'] = oz
        spec_group.attrs['VAOD'] = vaod
        spec_group.attrs['PWV'] = pwv
        
        
        wls = spec.lambdas
        flx = spec.data
        flx_err = spec.err
        
        d = spec_group.create_dataset("wls",data=wls,compression="gzip", compression_opts=9)
        d = spec_group.create_dataset("fls",data=flx,compression="gzip", compression_opts=9)
        d = spec_group.create_dataset("fls_err",data=flx_err,compression="gzip", compression_opts=9)
        
        c_dict = all_calspecs_sm[target_name]

        #smooth_data_np_convolve(sed,span)
        sed=np.interp(wls, c_dict["WAVELENGTH"]/10.,c_dict["FLUX"]*10.,left=1e-15,right=1e-15)
                      
        ratio = flx/tel.transmission(wls)/disp.transmission(wls)/sed
        ratio_err = flx/tel.transmission(wls)/disp.transmission(wls)/sed
       
        indexes = np.where(np.logical_and(wls>350.,wls<=1000.))[0]
       
        sel_wls = wls[indexes]
        sel_ratio = ratio[indexes]
        sel_ratio_airmas_corr = np.power(sel_ratio,am/spec.airmass)/(np.power(grey,am))
        sel_ratio_err = ratio_err[indexes]

        d = spec_group.create_dataset("wlr",data=sel_wls,compression="gzip", compression_opts=9)
        d = spec_group.create_dataset("ratio",data=sel_ratio,compression="gzip", compression_opts=9)
        d = spec_group.create_dataset("ratio_airmass1",data=sel_ratio_airmas_corr,compression="gzip", compression_opts=9)
        d = spec_group.create_dataset("ratio_err",data=sel_ratio_err,compression="gzip", compression_opts=9)
        count += 1
    group_name = "sim_spec"
    print(f">>>> create group {group_name}")
    sim_group =  hf.create_group(group_name)
   
    transm_sim1 = emul1.GetAllTransparencies(sel_wls,am,m_PWV,m_ozone,tau=m_VAOD)
    transm_sim2 = emul2.GetAllTransparencies(sel_wls,am,m_PWV,m_ozone,tau=m_VAOD)

    sim_group.attrs['ozone'] = m_ozone
    sim_group.attrs['VAOD'] = m_VAOD
    sim_group.attrs['PWV'] = m_PWV
    d = sim_group.create_dataset("wls",data=sel_wls,compression="gzip", compression_opts=9)
    d = sim_group.create_dataset("transm",data=transm_sim2,compression="gzip", compression_opts=9)   
    hf.close() 




# SMOOTHING

def smooth_data_convolve_my_average(arr, span):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")

    # The "my_average" part: shrinks the averaging window on the side that 
    # reaches beyond the data, keeps the other side the same size as given 
    # by "span"
    re[0] = np.average(arr[:span])
    for i in range(1, span + 1):
        re[i] = np.average(arr[:i + span])
        re[-i] = np.average(arr[-i - span:])
    return re

def smooth_data_np_average(arr, span):  # my original, naive approach
    return [np.average(arr[val - span:val + span + 1]) for val in range(len(arr))]

def smooth_data_np_convolve(arr, span):
    return np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")

def smooth_data_np_cumsum_my_average(arr, span):
    cumsum_vec = np.cumsum(arr)
    moving_average = (cumsum_vec[2 * span:] - cumsum_vec[:-2 * span]) / (2 * span)

    # The "my_average" part again. Slightly different to before, because the
    # moving average from cumsum is shorter than the input and needs to be padded
    front, back = [np.average(arr[:span])], []
    for i in range(1, span):
        front.append(np.average(arr[:i + span]))
        back.insert(0, np.average(arr[-i - span:]))
    back.insert(0, np.average(arr[-2 * span:]))
    return np.concatenate((front, moving_average, back))

def smooth_data_lowess(arr, span):
    x = np.linspace(0, 1, len(arr))
    return sm.nonparametric.lowess(arr, x, frac=(5*span / len(arr)), return_sorted=False)

def smooth_data_kernel_regression(arr, span):
    # "span" smoothing parameter is ignored. If you know how to 
    # incorporate that with kernel regression, please comment below.
    kr = KernelReg(arr, np.linspace(0, 1, len(arr)), 'c')
    return kr.fit()[0]

def smooth_data_savgol_0(arr, span):  
    return savgol_filter(arr, span * 2 + 1, 0)

def smooth_data_savgol_1(arr, span):  
    return savgol_filter(arr, span * 2 + 1, 1)

def smooth_data_savgol_2(arr, span):  
    return savgol_filter(arr, span * 2 + 1, 2)

def smooth_data_fft(arr, span):  # the scaling of "span" is open to suggestions
    w = fftpack.rfft(arr)
    spectrum = w ** 2
    cutoff_idx = spectrum < (spectrum.max() * (1 - np.exp(-span / 2000)))
    w[cutoff_idx] = 0
    return fftpack.irfft(w)


# Integrals


def fII0(wl,s):
    """
    Parameters:
    S : is the atmospheric transmission times the instrumental transmission
    wl :is the wavelength transmission

    return:
    II0 integral which is unitless
    """

    # clean
    indexes_sel = np.where(s>0.002)[0]
    wlmin = wl[indexes_sel].min()
    wlmax = wl[indexes_sel].max()

    indexes_sel = np.where(np.logical_and(wl>wlmin,wl<wlmax))[0]
    
    return np.trapz(s[indexes_sel]/wl[indexes_sel],wl[indexes_sel])
      
def fII1(wl,phi,wlb):
    """
    """
    return np.trapz(phi*(wl-wlb),wl)
  
def ZPT(wl,s):
    """
    Parameter:
    S : is the atmospheric transmission times the instrumental transmission
    wl :is the wavelength transmission

    return:
    The Zero point
    """
    return 2.5*np.log10(fII0(wl,s)) + ZPTconst

