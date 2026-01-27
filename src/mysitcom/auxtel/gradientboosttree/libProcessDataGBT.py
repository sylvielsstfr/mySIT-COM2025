"""Process data for Gradient Boosted Trees."""
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

    for f in the_filters:
        for t in the_targets:
            mask = (df[filter_col] == f) & (df[target_col] == t)
            data = df.loc[mask, feature_col]
            mean_data = data.mean()
            df[feature_col_out] = df[feature_col]/mean_data
    return df
