import pandas as pd
import numpy as np
import gc

# income_per_child
# days_with_same_bank
# number_loans_same_bank
# number_loans_any
# days_working_same_place
# days_working_total
# debt_income_ratio
# averages of different features
# amount paid of previous loans
# late payments
# credit limit exceeded
# loan completed


def agg_numeric(df, parent_var, df_name):
    """
    Groups and aggregates the numeric values in a child dataframe
    by the parent variable.

    Parameters
    --------
        df (dataframe):
            the child dataframe to calculate the statistics on
        parent_var (string):
            the parent variable used for grouping and aggregating
        df_name (string):
            the variable used to rename the columns

    Return
    --------
        agg (dataframe):
            a dataframe with the statistics aggregated by the `parent_var` for
            all numeric columns. Each observation of the parent variable will have
            one row in the dataframe with the parent variable as the index.
            The columns are also renamed using the `df_name`. Columns with all duplicate
            values are removed.

    """

    # Remove id variables other than grouping variable
    for col in df:
        if col != parent_var and 'SK_ID' in col:
            df = df.drop(columns=col)

    # Only want the numeric variables
    parent_ids = df[parent_var].copy()
    numeric_df = df.select_dtypes('number').copy()
    numeric_df[parent_var] = parent_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(parent_var).agg(['count', 'mean', 'max', 'min', 'sum'])

    # Need to create new column names
    columns = []

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        if var != parent_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns

    # Remove the columns with all redundant values
    _, idx = np.unique(agg, axis=1, return_index=True)
    agg = agg.iloc[:, idx]

    return agg


def agg_categorical(df, parent_var, df_name):
    """
    Aggregates the categorical features in a child dataframe
    for each observation of the parent variable.

    Parameters
    --------
    df : dataframe
        The dataframe to calculate the value counts for.

    parent_var : string
        The variable by which to group and aggregate the dataframe. For each unique
        value of this variable, the final dataframe will have one row

    df_name : string
        Variable added to the front of column names to keep track of columns


    Return
    --------
    categorical : dataframe
        A dataframe with aggregated statistics for each observation of the parent_var
        The columns are also renamed and columns with duplicate values are removed.

    """

    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('category'))

    # Make sure to put the identifying id on the column
    categorical[parent_var] = df[parent_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(parent_var).agg(['sum', 'count', 'mean'])

    column_names = []

    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['sum', 'count', 'mean']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))

    categorical.columns = column_names

    # Remove duplicate columns by values
    _, idx = np.unique(categorical, axis=1, return_index=True)
    categorical = categorical.iloc[:, idx]

    return categorical


# can be called by bureau and previous as they are direct children of app
def agg_child(df, parent_var, df_name):
    """Aggregate a child dataframe for each observation of the parent."""

    # Numeric and then categorical
    df_agg = agg_numeric(df, parent_var, df_name)
    df_agg_cat = agg_categorical(df, parent_var, df_name)

    # Merge on the parent variable
    df_info = df_agg.merge(df_agg_cat, on=parent_var, how='outer')

    # Remove any columns with duplicate values
    _, idx = np.unique(df_info, axis=1, return_index=True)
    df_info = df_info.iloc[:, idx]

    # memory management
    gc.enable()
    del df_agg, df_agg_cat
    gc.collect()

    return df_info


# can be called for the child classes of the child classes which is bureau_balance, cash, credit_card and installments
def agg_grandchild(df, parent_df, parent_var, grandparent_var, df_name):
    """
    Aggregate a grandchild dataframe at the grandparent level.

    Parameters
    --------
        df : dataframe
            Data with each row representing one observation

        parent_df : dataframe
            Parent table of df that must have the parent_var and
            the grandparent_var. Used only to get the grandparent_var into
            the dataframe after aggregations

        parent_var : string
            Variable representing each unique observation in the parent.
            For example, `SK_ID_BUREAU` or `SK_ID_PREV`

        grandparent_var : string
            Variable representing each unique observation in the grandparent.
            For example, `SK_ID_CURR`.

        df_name : string
            String for renaming the resulting columns.
            The columns are name with the `df_name` and with the
            statistic calculated in the column

    Return
    --------
        df_info : dataframe
            A dataframe with one row for each observation of the grandparent variable.
            The grandparent variable forms the index, and the resulting dataframe
            can be merged with the grandparent to be used for training/testing.
            Columns with all duplicate values are removed from the dataframe before returning.

    """

    # set the parent_var as the index of the parent_df for faster merges
    parent_df = parent_df[[parent_var, grandparent_var]].copy().set_index(parent_var)

    # Aggregate the numeric variables at the parent level
    df_agg = agg_numeric(df, parent_var, '%s_LOAN' % df_name)

    # Merge to get the grandparent variable in the data
    df_agg = df_agg.merge(parent_df,
                          on=parent_var, how='left')

    # Aggregate the numeric variables at the grandparent level
    df_agg_client = agg_numeric(df_agg, grandparent_var, '%s_CLIENT' % df_name)

    # Can only apply one-hot encoding to categorical variables
    if any(df.dtypes == 'category'):

        # Aggregate the categorical variables at the parent level
        df_agg_cat = agg_categorical(df, parent_var, '%s_LOAN' % df_name)
        df_agg_cat = df_agg_cat.merge(parent_df,
                                      on=parent_var, how='left')

        # Aggregate the categorical variables at the grandparent level
        df_agg_cat_client = agg_numeric(df_agg_cat, grandparent_var, '%s_CLIENT' % df_name)
        df_info = df_agg_client.merge(df_agg_cat_client, on=grandparent_var, how='outer')

        gc.enable()
        del df_agg, df_agg_client, df_agg_cat, df_agg_cat_client
        gc.collect()

    # If there are no categorical variables, then we only need the numeric aggregations
    else:
        df_info = df_agg_client.copy()

        gc.enable()
        del df_agg, df_agg_client
        gc.collect()

    # Drop the columns with all duplicated values
    _, idx = np.unique(df_info, axis=1, return_index=True)
    df_info = df_info.iloc[:, idx]

    return df_info


def calculate_correlation(df, col1, col2):
    return df[col1].corr(df[col2])




