# numpy and pandas for data manipulation
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, Imputer

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp
import statsmodels.api as sm


def save_pickle(path, data):
    # pickles the tokens dict
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print("File saved at ", path)


def load_pickle(path):
    # loads the tokens dict from directory
    with open(path, "rb") as f:
        return pickle.load(f)
    print("File loaded: ", path)


def load_training_data():
    app_train = pd.read_csv('../input/application_train.csv')
    print('Training data shape: ', app_train.shape)
    return app_train


def load_test_data():
    # Testing data features
    app_test = pd.read_csv('../input/application_test.csv')
    print('Testing data shape: ', app_test.shape)
    return app_test


def encode_binary_cols(app_train, app_test):
    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0
    encoded_cols = []
    # Iterate through the columns
    for col in app_train:
        if app_train[col].dtype == 'object':
            # If 2 or fewer unique categories (a nan will count as a category)
            if len(list(app_train[col].unique())) <= 2:
                # Train on the training data
                le.fit(app_train[col])
                # Transform both training and testing data
                app_train[col] = le.transform(app_train[col])
                app_test[col] = le.transform(app_test[col])
                encoded_cols.append(col)

                # Keep track of how many columns were label encoded
                le_count += 1
    return app_train, app_test


def one_hot_encode(app_train, app_test):
    # one-hot encoding of categorical variables
    # Dummy encoding will not create a column for nans
    app_train = pd.get_dummies(app_train)
    app_test = pd.get_dummies(app_test)

    print("ONE HOT ENCODED")
    print('Training Features shape: ', app_train.shape)
    print('Testing Features shape: ', app_test.shape)
    return app_train, app_test


def align_data(app_train, app_test):
    # ALIGN TEST AND TRAIN DATAFRAMES SO COLUMNS MATCH
    train_labels = app_train['TARGET']

    # Align the training and testing data, keep only columns present in both dataframes
    app_train, app_test = app_train.align(app_test, join='inner', axis=1)

    # Add the target back in
    app_train['TARGET'] = train_labels

    print("ALIGNED:")
    print('Training Features shape: ', app_train.shape)
    print('Testing Features shape: ', app_test.shape)
    return app_train, app_test, train_labels


def remove_days_employed_anomaly(app_train, app_test):
    # DEALING WITH ANOMALOUS DATA IN 'DAYS_EMPLOYED' COL

    # Create an anomalous flag column
    app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
    # Replace the anomalous values with nan
    app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

    app_train['DAYS_EMPLOYED'].plot.hist(title='Days Employment Histogram')
    plt.xlabel('Days Employment')

    app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
    app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    print('There were %d anomalies in the test data out of %d entries' % (
        app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))
    return app_train, app_test


def remove_missing_cols(app_train, app_test, thr=0.68):
    app_train = app_train.loc[:,
                app_train.isnull().mean() < thr]  # remove all columns with more than x% missing values
    print('Training Features shape: ', app_train.shape)
    print('Testing Features shape: ', app_test.shape)

    # ALIGN TEST AND TRAIN DATAFRAMES SO COLUMNS MATCH
    train_labels = app_train['TARGET']
    # Align the training and testing data, keep only columns present in both dataframes
    app_train, app_test = app_train.align(app_test, join='inner', axis=1)
    # Add the target back in
    app_train['TARGET'] = train_labels

    print('Training Features shape: ', app_train.shape)
    print('Testing Features shape: ', app_test.shape)
    return app_train, app_test


def mean_impute(df):
    return df.fillna(df.mean())


def normalise(train, test):
    # MAKE SURE SK_CURR_ID AND TARGET have been dropped
    scaler = MinMaxScaler(feature_range=(0, 1))  # Scale each feature to 0-1
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    print("NORMALISED:")
    print('Training data shape: ', train.shape)
    print('Testing data shape: ', test.shape)
    return train, test


def normalise_and_impute(app_train, app_test, impute_strategy='mean'):
    # Drop the target from the training data
    if 'TARGET' in app_train:
        train = app_train.drop(columns=['TARGET'])
    else:
        train = app_train.copy()
    train = train.drop(columns=['SK_ID_CURR'])  #

    test = app_test.copy()
    test = test.drop(columns=['SK_ID_CURR'])  #

    # Feature names
    features = list(train.columns)

    # Median imputation of missing values
    imputer = Imputer(strategy=impute_strategy)
    # Fit on the training data
    imputer.fit(train)
    # Transform both training and testing data
    train = imputer.transform(train)
    test = imputer.transform(test)  ### test was app_test

    # Normalise
    scaler = MinMaxScaler(feature_range=(0, 1))  # Scale each feature to 0-1
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    print("IMPUTED AND NORMALISED")
    print('Training data shape: ', train.shape)
    print('Testing data shape: ', test.shape)
    return train, test, features


def create_and_save_submission(app_test, predictions, save_path):
    # Submission dataframe
    submit = app_test[['SK_ID_CURR']]
    submit['TARGET'] = predictions

    # Save the submission to a csv file
    submit.to_csv(save_path, index=False)
    print("Predictions saved to: ", save_path)


def cross_val_roc_curve(train_X, train_Y, classifier):
    # From https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html

    # ROC AUC with stratified cross validation
    X = train_X
    y = train_Y

    cv = StratifiedKFold(n_splits=6, shuffle=True)
    tprs = []  # true positive rate scores
    aucs = []  # area under curve scores
    mean_fpr = np.linspace(0, 1, 100)  # mean false positive rates

    i = 0
    # train and test for each fold
    for train_sample, test_sample in cv.split(X, y):
        probas_ = classifier.fit(X[train_sample], y[train_sample]).predict_proba(X[test_sample])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test_sample], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        print("Run {} AUC socre: {}".format(i, roc_auc))

        '''Everything below this point is just for the plot'''
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    # plot roc curve for fold
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    # caluclate and plot mean auc
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    # plot standard deviation area
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    # add labels to plot
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    print("Avg ROC AUC score: {}".format(np.mean(aucs)))


def feature_aic_bic(app_train, feature_name: str):
    # calculates the aic and bic values between the target and a column
    # http://www.differencebetween.net/miscellaneous/difference-between-aic-and-bic/

    data = app_train.copy()
    data = data[[feature_name, "TARGET"]]

    # Median imputation of missing values
    imputer = Imputer(strategy='mean')
    imputer.fit(data)
    imputed_data = imputer.transform(data)
    data[feature_name] = imputed_data

    data["intercept"] = 1.0
    logit = sm.Logit(data["TARGET"], data[feature_name])
    result = logit.fit()

    print("Selected Feature", feature_name)
    result.summary2()  # uncomment for full summary
    # print("AIC", result.aic)
    # print("BIC", result.bic)


def imputed_col_aic(data, feature_name):
    #     data["intercept"] = 1.0
    logit = sm.Logit(data["TARGET"], data[feature_name])
    result = logit.fit()
    # print("AIC", result.aic)
    return result.aic
