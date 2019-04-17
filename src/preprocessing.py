from utils import*

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, Imputer


def load_data(filename):
    return pd.read_csv(filename)


def one_hot_encoding(data):
    return pd.get_dummies(data)


def remove_days_employed_anomaly(data):
    """
    Removes the 365243 anomaly (from DAYS_EMPLOYED) by replacing it with a nan. Also creates a binary feature indicating
    at which rows this anomalous value occurred
    :param train: training dataframe containing the DAYS_EMPLOYED columns
    :return: train dataframes with the anomaly removed
    """
    # Make anomalous flag column
    data['DAYS_EMPLOYED_ANOM'] = data["DAYS_EMPLOYED"] == 365243
    # Replace anomalous values with nan
    data['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

    return data


def encode_binary_cols(data):
    """
    Applies label encoding to the train and test dataframes.
    Will only do this if there are more than 2 labels (nan is considered a category)

    :param data: training data
    :return: train data with label encoded columns
    """
    le = LabelEncoder()
    encoded_cols = []
    for col in data:
        if data[col].dtype == 'object':
            # If 2 or fewer unique categories (a nan will count as a category)
            if len(list(data[col].unique())) <= 2:
                # Train on the training data
                le.fit(data[col])
                # Transform both training and testing data
                data[col] = le.transform(data[col])
                encoded_cols.append(col)
    print("Label encoded columns", encoded_cols)

    return data


def align_data(train, test, verbose=True):
    """
    Aligns the training and test dataframes so that all columns in testing are also in training (bar TARGET)
    :param train: training dataframe
    :param test: test dataframe
    :return: aligned dataframes
    """
    train_labels = train['TARGET']
    train, test = train.align(test, join='inner', axis=1)
    train['TARGET'] = train_labels

    if verbose:
        print("AFTER ALIGNMENT:")
        print('Training Features shape: ', train.shape)
        print('Testing Features shape: ', test.shape)

    return train, test


def remove_missing_cols(data, thr=0.68):
    """
    Removes columns from the training data with over a certain threshold of missing values. The test data is then
    aligned
    :param data: dataframe
    :param test: dataframe
    :param thr: If less than the threshold the column will be removed
    :return:
    """
    print("Removing columns with {} proportion of missing values".format(thr))
    data = data.loc[:,
           data.isnull().mean() < thr]  # remove all columns with more than x% missing values
    # align_data(train, test, verbose=False)

    print("AFTER REMOVING MISSING COLS (and aligning):")
    print('Training Features shape: ', data.shape)
    # print('Testing Features shape: ', test.shape)
    return data


def mean_imputation(data):
    """
    Applies mean imputation to all columns with missing values in the the train and test data
    The imputer is fitted on the training data only and applied to both train and test, meaning that both dataframes
    require to be aligned beforehand
    :param data: Training data
    :param test: Test data
    :return: Dataframes of the train and test with all columns mean imputed
    """
    imputer = Imputer(strategy='mean')
    # Fit on the training data
    imputer.fit(data)
    # Transform both training and testing data
    data[data.columns] = imputer.transform(data[data.columns])
    # test[test.columns] = imputer.transform(test[test.columns])

    print("AFTER MEAN IMPUTATION:")
    print('Training data shape: ', data.shape)
    # print('Testing data shape: ', test.shape)

    return data


def normalise(data):
    """
    Normalises features in the test and train dataframes to be between 0-1
    MAKE SURE SK_CURR_ID AND TARGET have been dropped!
    :param data: training dataframe
    :param test: test dataframe
    :return: normalised train and test dataframes
    """
    assert "TARGET" not in data, "TARGET column should be dropped in train"
    assert "SK_ID_CURR" not in data, "SK_ID_CURR column should be dropped in train"
    # assert "SK_ID_CURR" not in test, "SK_ID_CURR column should be dropped in test"

    scaler = MinMaxScaler(feature_range=(0, 1))  # Scale each feature to 0-1
    scaler.fit(data)
    data[data.columns] = scaler.transform(data[data.columns])
    # test[test.columns] = scaler.transform(test[test.columns])

    print("AFTER NORMALISATION:")
    print('Training data shape: ', data.shape)
    # print('Testing data shape: ', test.shape)
    return data
