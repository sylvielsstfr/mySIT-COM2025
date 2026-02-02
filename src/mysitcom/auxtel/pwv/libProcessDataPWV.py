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


