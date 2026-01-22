""" Tools for astro """
from astropy.time import Time
from datetime import datetime
from astropy.coordinates import EarthLocation, AltAz, get_sun
import numpy as np
import astropy.units as u
import pandas as pd

# Observation site
site_lsst = EarthLocation.of_site("Cerro Pachon")


def get_astronomical_midnight(location: EarthLocation, date, n_grid=1000):
    """
    Transit inférieur du Soleil (min altitude) pour date & site avec astropy pur.
    
    Parameters
    ----------
    location : EarthLocation
        Site d’observation (lat, lon, hauteur)
    date : str or Time
        Date de référence (ex: "2025-09-23")
    n_grid : int
        Nombre de points sur 24h à évaluer pour estimer le minimum
    
    Returns
    -------
    Time
        Temps UTC quand le Soleil a l’altitude minimale
    """


    if isinstance(date, datetime):
        t = Time(date, scale="utc")
    elif hasattr(date, "strftime"):  # ex: datetime.date
        t = Time(date.strftime("%Y-%m-%d"), scale="utc")
    elif not isinstance(date, Time):
        t = Time(date, scale="utc")
    else:
        t = date
    
    date = t + 1 * u.day
    
    # Définir intervalle de ~24h autour de la date
    t0 = date - 12  * u.hour
    t1 = date + 12  * u.hour
    
    # Grille de temps
    times = Time(np.linspace(t0.jd, t1.jd, n_grid), format='jd', scale='utc')
    
    # Position du Soleil
    sun = get_sun(times)
    
    # AltAz
    aa = AltAz(obstime=times, location=location)
    sun_altaz = sun.transform_to(aa)
    
    # Trouver index du min
    idx_min = np.argmin(sun_altaz.alt)
    t_min = times[idx_min]
    return t_min.to_datetime()

#---------------------------------------------------------------

def GetNightMidnightsDict(df,nightobs_col = "nightObs"):
    """
    input:
      df_spec the dataframe for spectroscopy summary results
    output:
      the dict of midnights
    """
    
    Dt = pd.Timedelta(minutes=30)
    d = {}
    list_of_nightobs = df[nightobs_col].unique()
    for nightobs in list_of_nightobs:
        nightstr = datetime.strptime(str(nightobs), "%Y%m%d")
        midnight = get_astronomical_midnight(site_lsst, nightstr.date())
        d[nightobs] = midnight
        
    return d
