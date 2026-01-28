"""Process data for Gradient Boosted Trees."""
import numpy as np
import pandas as pd

def normalize_column_data(df,target_col,filter_col,feature_col,ext="norm"):
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
            #data = df.loc[mask, feature_col]
            df_data = df[mask]
            mean_data = df_data[feature_col].mean()
            df_data[feature_col_out] = df_data[feature_col]/mean_data
            all_df.append(df_data)
            df_out.loc[len(df_out)] = {target_col: t, filter_col: f, feature_col_mean: mean_data}

    df_merge = pd.concat(all_df)
    df_merge = df_merge.sort_values(by="id", ascending=True)

    return df_merge,df_out




