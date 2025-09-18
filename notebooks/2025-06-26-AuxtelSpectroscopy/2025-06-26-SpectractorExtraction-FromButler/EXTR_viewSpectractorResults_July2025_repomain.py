#!/usr/bin/env python
# coding: utf-8

# # Extract from Butler
# 
# - from Tuto of Corentin R on Spectractor, May 28th 2025
# - After adapting path of Butler in atmospec/spectraction.py
# - adaptation : Sylvie Dagoret-Campagne
# - date : 2025-07-03

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.io import fits
import getCalspec
import os


# In[ ]:


from lsst.summit.utils.utils import checkStackSetup
checkStackSetup()


# In[ ]:


import lsst.daf.butler as dafButler

repo = "/repo/main"
butler = dafButler.Butler(repo)
registry = butler.registry


for c in sorted(registry.queryCollections()):
    if "dagoret" in c:
        print(c)


# Find collection in Butler /repo/embargo
#my_collection = ['u/dagoret/auxtel_run_20250625a']
#my_collection = ['u/dagoret/auxtel_run_20250702b']
my_collection = ['u/dagoret/auxtel_run_20250702a']


# save extraction
#file_save = "auxtel_atmosphere_20250625a_v1.npy"
#file_save = "auxtel_atmosphere_20250702b_repomain_v1.npy"
file_save = "auxtel_atmosphere_20250702a_repomain_v1.npy"



datasetRefs = registry.queryDatasets(datasetType='spectractorSpectrum', collections=my_collection, where= "instrument='LATISS'")
where = "instrument='LATISS'" 
records = list(butler.registry.queryDimensionRecords('visit', datasets='spectractorSpectrum', where=where,  collections=my_collection))
refs = list(set(butler.registry.queryDatasets('spectractorSpectrum',  where=where,  collections=my_collection)))
len(records)




for i, r in enumerate(records):

    print(f"============= ({i}) ============datasetType = spectraction ============================================")
    print("fullId..................:",r.id)
    print("seq_num..................:",r.seq_num)
    print("day_obs..................:",r.day_obs)
    print("target..................:",r.target_name)
    print("filt+disp..................:",r.physical_filter)

    # spec = butler.get('spectractorSpectrum', visit=r.id, detector=0, collections=my_collection, instrument='LATISS')
    
    if i>5:
        break




# ## Load one spectrum




print(butler.registry.getDatasetType('spectrumLibradtranFitParameters'))


#


# for i in range(len(refs_noerrorsed)):
for i in range(20):
    try:        
        p = butler.get('spectrumLibradtranFitParameters', visit=refs_noerrorsed[i].dataId["visit"], collections=my_collection, detector=0, instrument='LATISS')
        err = p["ozone [db]"]
    except:
        pass




# for i in range(len(refs_noerrorsed)):
for i in range(20):
    try:        
        p = butler.get('spectrogramLibradtranFitParameters', visit=refs_noerrorsed[i].dataId["visit"], collections=my_collection, detector=0, instrument='LATISS')
        err = p["ozone [db]"]
    except:
        pass
    


# In[ ]:


dataId = {"day_obs": 20220316, "seq_num": 330, 'instrument':'LATISS',"detector": 0}

20220316

spec= butler.get('spectractorSpectrum',dataId,collections=my_collection)
p = butler.get('spectrumLibradtranFitParameters',dataId,collections=my_collection)





if not(os.path.isfile(file_save)):
    # see here an efficient way to access FITS headers: https://lsstc.slack.com/archives/CBV7K0DK6/p1700250222827499
    params_spectrum = []
    params_spectrogram = []
    headers = []
    
    def from_ref_to_dataId(ref):
        dataId = {'day_obs': ref.dataId["day_obs"], 'seq_num': int(str(ref.dataId["visit"])[8:]), 'instrument': 'LATISS', 'detector': 0}
        return dataId
    
    for ref in tqdm(sorted(refs, key=lambda x: x.dataId["visit"])[::]):
        try:
            spec = butler.get('spectractorSpectrum', visit=ref.dataId["visit"], collections=my_collection, detector=0, instrument='LATISS')
            headers.append(spec.header)
            p = butler.get('spectrumLibradtranFitParameters', visit=ref.dataId["visit"], collections=my_collection, detector=0, instrument='LATISS')
            params_spectrum.append(p)
            p = butler.get('spectrogramLibradtranFitParameters', visit=ref.dataId["visit"], collections=my_collection, detector=0, instrument='LATISS')
            params_spectrogram.append(p)
        except (AttributeError,ValueError,LookupError):
            print("Skip", ref.dataId["visit"])
            continue





if not(os.path.isfile(file_save)):
    columns_spectrum = ["id"]
    
    for h in headers[0]:
        if "COMMENT" in h or "EXTNAME" in h: continue
        if "LBDAS_T" in h or "PSF_P_T" in h or "AMPLIS_T" in h: continue
        if "UNIT" in h: continue
        if "SIMPLE" in h: continue
        columns_spectrum.append(h)
     
    columns_spectrogram_bestfit = []
    for key in params_spectrogram[0].labels:
        columns_spectrogram_bestfit.append(key)
        columns_spectrogram_bestfit.append(key+"_err")
    
    columns_spectrum_bestfit = []
    for key in params_spectrum[0].labels:
        columns_spectrum_bestfit.append(key)
        columns_spectrum_bestfit.append(key+"_err")
    
    min_index = 0
    max_index = np.inf

    #df1 is header info
    df1 = pd.DataFrame(columns=columns_spectrum)
    
    for k, header in enumerate(headers):
        # if k > 40: break
        n = records[k].id
        if n < min_index or n > max_index: continue
        row = {"id": n}
        for h in header:
            if h in columns_spectrum:
                row[h] = header[h]
        df1.loc[len(df1)] = row

    #df2 is spectrogram     spectrogram best fit
    df2 = pd.DataFrame(columns=columns_spectrogram_bestfit)
    
    for k, p in enumerate(params_spectrogram):
        n = records[k].id
        if n < min_index or n > max_index: continue
        row = {"id": n}
        for i, key in enumerate(p.labels):
            row[key] = p.values[i]
            row[key+"_err"] = p.err[i]
        df2.loc[len(df2)] = row

    # df3 is spectrum best fit    
    df3 = pd.DataFrame(columns=columns_spectrum_bestfit)

    
    for k, p in enumerate(params_spectrum):
        n = records[k].id
        if n < min_index or n > max_index: continue
        row = {"id": n}
        for i, key in enumerate(p.labels):
            row[key] = p.values[i]
            row[key+"_err"] = p.err[i]
        df3.loc[len(df3)] = row

    # merge header with spectrogram
    df = pd.merge(df1, df2, left_index=True, right_index=True)
    # merge (header-spectrogram with spectrum)
    df = pd.merge(df, df3, left_index=True, right_index=True)
    df.set_index('DATE-OBS', inplace=True)
    df.index = pd.to_datetime(df.index, format="ISO8601") #['DATE-OBS'])
    df.sort_index(inplace=True)
    
    rec = df.to_records()
    np.save(file_save, rec)



rec = np.load(file_save, allow_pickle=True)
df = pd.DataFrame(rec)
pd.set_option('display.max_columns', None)
print(rec.shape)



