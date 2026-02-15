"""Process data for Gradient Boosted Trees."""
import numpy as np
import pandas as pd


def normalize_column_data_bytarget_byfilter(df,target_col,filter_col,feature_col,ext="norm"):
    """
    Docstring pour normalize_data
    :param df: Pandas dataframe
    :param target_col: name of columns target in dataframe
    :param filter_col: name of columns filter in dataframe
    :param feature_col: name of columns feature in dataframe
    :param ext: extension to add to new column name
    :return: dataframe with new normalized feature column
    """

    the_filters = df[filter_col].unique()
    the_targets = df[target_col].unique()
    feature_col_out =f"{feature_col}_{ext}"
    feature_col_mean =f"{feature_col}_mean"

    all_df = []
    df_out =  pd.DataFrame(columns=[target_col ,filter_col,feature_col_mean])

    for f in the_filters:
        for t in the_targets:
            mask = (df[filter_col] == f) & (df[target_col] == t)
            df_data = df[mask]
            mean_data = df_data[feature_col].mean()
            df_data[feature_col_out] = df_data[feature_col]/mean_data
            all_df.append(df_data)
            df_out.loc[len(df_out)] = {target_col: t, filter_col: f, feature_col_mean: mean_data}

    df_merge = pd.concat(all_df)
    df_merge = df_merge.sort_values(by="id", ascending=True)

    return df_merge,df_out



def normalize_column_data_bytarget_byfilter_bymethod(df,target_col,filter_col,feature_col,
                                            ext="norm",method="mean"):
    """
    Docstring pour normalize_data
    :param df: Pandas dataframe
    :param target_col: name of columns target in dataframe
    :param filter_col: name of columns filter in dataframe
    :param feature_col: name of columns feature in dataframe
    :param ext: extension to add to new column name
    :param method: normalization method (mean or median)
    :return: dataframe with new normalized feature column
    """

    the_filters = df[filter_col].unique()
    the_targets = df[target_col].unique()
    feature_col_out =f"{feature_col}_{ext}"
    feature_col_method =f"{feature_col}_{method}"

    all_df = []
    df_out =  pd.DataFrame(columns=[target_col ,filter_col,feature_col_method])

    for f in the_filters:
        for t in the_targets:
            mask = (df[filter_col] == f) & (df[target_col] == t)
            df_data = df[mask]

            if method=="mean":
                mdata = df_data[feature_col].mean()
                df_data[feature_col_out] = df_data[feature_col]/mdata
            elif method=="median":
                mdata = df_data[feature_col].median()

            df_data[feature_col_out] = df_data[feature_col]/mdata

            all_df.append(df_data)
            df_out.loc[len(df_out)] = {target_col: t, filter_col: f, feature_col_method: mdata}

    df_merge = pd.concat(all_df)
    df_merge = df_merge.sort_values(by="id", ascending=True)

    return df_merge,df_out

def shiftaverage_column_data_byfilter(df,night_col,filter_col,feature_col,ext="shift"):
    """
    Compute shift relative to night average
    :param df: Pandas dataframe
    :param night_col: name of columns night in dataframe
    :param filter_col: name of columns filter in dataframe
    :param feature_col: name of columns feature in dataframe
    :param ext: extension to add to new column name
    :return: dataframe with new normalized feature column
    """

    the_filters = df[filter_col].unique()
    the_nights = df[night_col].unique()
    feature_col_out =f"{feature_col}_{ext}"
    #feature_col_mean =f"{feature_col}_mean"


    all_df = []
    #df_out =  pd.DataFrame(columns=[target_col ,filter_col,feature_col_mean])

    for f in the_filters:
        for night in the_nights:
            mask = (df[filter_col] == f) & (df[night_col] == night)
            df_data = df[mask]
            mean_data = df_data[feature_col].mean()
            df_data[feature_col_out] = df_data[feature_col] - mean_data
            all_df.append(df_data)
            #df_out.loc[len(df_out)] = {night_col: night, filter_col: f, feature_col_mean: mean_data}

    df_merge = pd.concat(all_df)
    df_merge = df_merge.sort_values(by="id", ascending=True)

    return df_merge

#-----

def pwv_deviation_from_linear_interp(
    df,
    night_col,
    filter_col,
    target_col,
    time_col,
    pwv_col,
    suffix="lininterp"
):
    """
    For each (target, night, filter), compute for each observation:
      1) deviation from linear temporal interpolation of PWV
         between previous and next observations
      2) total time separation between previous and next observations

    Returns dataframe with two new columns:
      - pwv_<suffix>
      - dt_<suffix>
    """

    df = df.copy()

    pwv_dev_col = f"{pwv_col}_{suffix}"
    dt_col = f"dt_{suffix}"

    df[pwv_dev_col] = np.nan
    df[dt_col] = np.nan

    # Group by target / night / filter
    grouped = df.groupby([target_col, night_col, filter_col])

    for _, g in grouped:

        # Sort by time
        g = g.sort_values(time_col)

        t = g[time_col].values
        pwv = g[pwv_col].values

        if len(g) < 3:
            continue  # not enough points for interpolation

        # Previous and next
        t_prev = t[:-2]
        t_curr = t[1:-1]
        t_next = t[2:]

        pwv_prev = pwv[:-2]
        pwv_curr = pwv[1:-1]
        pwv_next = pwv[2:]

        # Linear interpolation
        pwv_interp = pwv_prev + (pwv_next - pwv_prev) * (
            (t_curr - t_prev) / (t_next - t_prev)
        )

        pwv_dev = pwv_curr - pwv_interp
        dt_tot = t_next - t_prev

        idx = g.index[1:-1]

        df.loc[idx, pwv_dev_col] = pwv_dev
        df.loc[idx, dt_col] = dt_tot

    return df




def pwv_deviation_from_linear_interp_datetime(
    df,
    night_col,
    filter_col,
    target_col,
    time_col,
    pwv_col,
    time_unit="s",          # "s", "min", "h", "d"
    suffix="repeat"
):
    """
    Same as pwv_deviation_from_linear_interp but with time_col
    as pd.datetime64.

    time_unit controls the unit of dt output.
    """

    df = df.copy()

    pwv_dev_col = f"{pwv_col}_{suffix}"
    dt_col = f"dt_{suffix}"

    df[pwv_dev_col] = np.nan
    df[dt_col] = np.nan

    grouped = df.groupby([target_col, night_col, filter_col])

    for _, g in grouped:

        g = g.sort_values(time_col)

        t = g[time_col].values
        pwv = g[pwv_col].values

        if len(g) < 3:
            continue

        t_prev = t[:-2]
        t_curr = t[1:-1]
        t_next = t[2:]

        # Timedelta objects
        dt_prev = t_curr - t_prev
        dt_tot = t_next - t_prev

        # Convert to float in chosen unit
        dt_prev_val = pd.to_timedelta(dt_prev).astype("timedelta64[s]")
        dt_tot_val = pd.to_timedelta(dt_tot).astype("timedelta64[s]")

        unit_scale = {
            "s": 1.0,
            "min": 60.0,
            "h": 3600.0,
            "d": 86400.0,
        }[time_unit]

        dt_prev_val = dt_prev_val / unit_scale
        dt_tot_val = dt_tot_val / unit_scale

        pwv_prev = pwv[:-2]
        pwv_curr = pwv[1:-1]
        pwv_next = pwv[2:]

        pwv_interp = pwv_prev + (pwv_next - pwv_prev) * (
            dt_prev_val / dt_tot_val
        )

        pwv_dev = pwv_curr - pwv_interp

        idx = g.index[1:-1]

        df.loc[idx, pwv_dev_col] = pwv_dev
        df.loc[idx, dt_col] = dt_tot_val

        df["dt_col"] = pd.to_timedelta(df[dt_col], unit=time_unit)

    return df


def pwv_next_prev_difference_datetime(
    df,
    night_col,
    filter_col,
    target_col,
    time_col,
    pwv_col,
    time_unit="s",          # "s", "min", "h", "d"
    suffix="nextprev"
):
    """
    Compute PWV(next) - PWV(previous) and time(next) - time(previous),
    using datetime columns. Values are assigned to the central row.
    """

    df = df.copy()

    pwv_diff_col = f"{pwv_col}_{suffix}"
    dt_col = f"dt_{suffix}"

    df[pwv_diff_col] = np.nan
    df[dt_col] = np.nan

    grouped = df.groupby([target_col, night_col, filter_col])

    unit_scale = {
        "s": 1.0,
        "min": 60.0,
        "h": 3600.0,
        "d": 86400.0,
    }[time_unit]

    for _, g in grouped:

        g = g.sort_values(time_col)

        if len(g) < 3:
            continue

        t = g[time_col].values
        pwv = g[pwv_col].values

        t_prev = t[:-2]
        t_next = t[2:]

        pwv_prev = pwv[:-2]
        pwv_next = pwv[2:]

        # Time difference
        dt = t_next - t_prev
        dt_val = pd.to_timedelta(dt).astype("timedelta64[s]") / unit_scale

        # PWV difference
        pwv_diff = pwv_next - pwv_prev

        idx = g.index[1:-1]

        df.loc[idx, pwv_diff_col] = pwv_diff
        df.loc[idx, dt_col] = dt_val

    # Optional: store dt as timedelta again
    df[dt_col] = pd.to_timedelta(df[dt_col], unit=time_unit)

    return df

def compute_atmparam_stats_per_filter(
    df,
    filter_col,
    param_col,
):
    """
    Compute statistics on param_col per filter.

    Returns
    -------
    pandas.DataFrame
        index   : filter
        columns : mean, median, sigma, sigma_mad, sigma_iqr
    """

    data = df.copy()

    # ----------------------------
    # data to compute
    # ----------------------------
    data = data[[filter_col, param_col]].dropna()

    results = []

    # ----------------------------
    # compute per filter
    # ----------------------------
    for f, subdf in data.groupby(filter_col):
        sub = subdf[param_col].values

        mean = np.mean(sub)
        median = np.median(sub)
        sigma = np.std(sub, ddof=1)

        # IQR-based sigma
        q25, q75 = np.percentile(sub, [25, 75])
        sigma_iqr = (q75 - q25) / 1.349

        # MAD-based sigma
        mad = np.median(np.abs(sub - median))
        sigma_mad = 1.4826 * mad

        results.append(
            dict(
                filter=f,
                mean=mean,
                median=median,
                sigma=sigma,
                sigma_mad=sigma_mad,
                sigma_iqr=sigma_iqr,
            )
        )

    # ----------------------------
    # Build DataFrame
    # ----------------------------
    df_stats = (
        pd.DataFrame(results)
        .set_index("filter")
        .sort_index()
    )

    return df_stats.T

#----
def compute_all_performances_on_diffpwv_pwvrepinterp_pwvrepdiff(df):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    # statistics on PWV_ram-PWV_rum
    df_pwvdiff = compute_atmparam_stats_per_filter(
        df,
        filter_col="FILTER",
        param_col = "diff_PWV")

    ## With PWV repeatability with time interpolation
    ### Calculate Repeatability with time interpolation for Spectrogram
    df = pwv_deviation_from_linear_interp_datetime(
        df,
        night_col="nightObs",
        filter_col="FILTER",
        target_col="TARGET",
        time_col="Time",
        pwv_col="PWV [mm]_ram",
        suffix="repeatinterp_ram",
        time_unit="min",
    )
    ### Calculate Repeatability with time interpolation for Spectrum
    df = pwv_deviation_from_linear_interp_datetime(
        df,
        night_col="nightObs",
        filter_col="FILTER",
        target_col="TARGET",
        time_col="Time",
        pwv_col="PWV [mm]_rum",
        suffix="repeatinterp_rum",
        time_unit="min",
    )
    #### Process some columns format
    df["dt_repeatinterp_ram"] = pd.to_timedelta(df["dt_repeatinterp_ram"])
    df["dt_repeatinterp_rum"] = pd.to_timedelta(df["dt_repeatinterp_rum"])

    df_pwv_repeatinterp_ram = compute_atmparam_stats_per_filter(
        df,
        filter_col="FILTER",
        param_col = "PWV [mm]_ram_repeatinterp_ram")

    df_pwv_repeatinterp_rum = compute_atmparam_stats_per_filter(
        df,
        filter_col="FILTER",
        param_col = "PWV [mm]_rum_repeatinterp_rum")

    df = df.dropna(subset=["PWV [mm]_ram_repeatinterp_ram"])
    df = df.dropna(subset=["PWV [mm]_rum_repeatinterp_rum"])

    ### PWV repeatability with PWV difference next-previous
    #### Calculate PWV repeatability with PWV difference next-previous in Spectrogram
    df = pwv_next_prev_difference_datetime(
        df,
        night_col="nightObs",
        filter_col="FILTER",
        target_col="TARGET",
        time_col="Time",
        pwv_col="PWV [mm]_ram",
        suffix="repeatdiff_ram",
        time_unit="min",
    )

    #### Calculate PWV repeatability with PWV difference next-previous  in Spectrum
    df = pwv_next_prev_difference_datetime(
        df,
        night_col="nightObs",
        filter_col="FILTER",
        target_col="TARGET",
        time_col="Time",
        pwv_col="PWV [mm]_rum",
        suffix="repeatdiff_rum",
        time_unit="min",
    )

    ##### Process some columns format
    df["dt_repeatdiff_ram"] = pd.to_timedelta(df["dt_repeatdiff_ram"])
    df["dt_repeatdiff_rum"] = pd.to_timedelta(df["dt_repeatdiff_rum"])
    df["dt_repeatdiff_ram_min"] = df["dt_repeatdiff_ram"].dt.total_seconds() / 60
    df["dt_repeatdiff_rum_min"] = df["dt_repeatdiff_rum"].dt.total_seconds() / 60


    df_pwv_repeatdiff_ram = compute_atmparam_stats_per_filter(
        df,
        filter_col="FILTER",
        param_col = "PWV [mm]_ram_repeatdiff_ram")
    df_pwv_repeatdiff_rum = compute_atmparam_stats_per_filter(
        df,
        filter_col="FILTER",
        param_col = "PWV [mm]_rum_repeatdiff_rum")

    ### Combine statistics of histograms
    # 1. define suffixes order and name
    suffixes = ['_diff', '_repinterp_ram', '_repinterp_rum', '_repdiff_ram', '_repdiff_rum']

    # 2. combine the stat dataframe un the same order as the suffixes
    all_df_stats = [ df_pwvdiff, df_pwv_repeatinterp_ram , df_pwv_repeatinterp_rum, df_pwv_repeatdiff_ram,df_pwv_repeatdiff_rum]

    # 3. Dictionnary suffix -> dataframe
    df_dict = dict(zip(suffixes, all_df_stats))

    # 4.  concatenate horizontally
    df_final = pd.concat(df_dict.values(), axis=1)

    # 5. Rename columns by adding suffixes
    # On boucle sur le dictionnaire pour construire les nouveaux noms
    new_columns = []
    for suffix, df in df_dict.items():
        new_columns.extend([f"{col}{suffix}" for col in df.columns])

    df_final.columns = new_columns
    return df_final.T, df
