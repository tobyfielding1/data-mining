import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, Imputer

warnings.filterwarnings('ignore')  # supress warnings
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp
import statsmodels.api as sm

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from scipy.stats import boxcox

import csv

from timeit import default_timer as timer

from tqdm import tqdm

import random

# Modeling
import lightgbm as lgb

# Parameter tuning
from hyperopt import fmin
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import STATUS_OK

from functools import partial


# do basic pre-processing
def pre_process(train_data, test_data):
    
    train_data, test_data = remove_days_employed_anomaly(train_data, test_data)
    
    train_data, test_data = encode_binary_cols(train_data, test_data)
    train_data, test_data = one_hot_encode(train_data, test_data)
    train_data, test_data = align_data(train_data, test_data)
    train_data, test_data = remove_missing_cols(train_data, test_data, thr=0.68)
    
    train_data, test_data = align_data(train_data, test_data)
    
    # need to drop TARGET and SK_ID_CURR before normalisation 
    train_data.drop(columns=['TARGET', 'SK_ID_CURR'],inplace=True)
    test_data.drop(columns=['SK_ID_CURR'], inplace= True)
    train_data, test_data = mean_imputation(train_data, test_data) # may be a bit slow
    train_data, test_data = normalise(train_data, test_data)

    return train_data, test_data


def save_pickle(path, data):
    """
    pickles the tokens dict
    :param path: paths to save at
    :param data: data to pickle (can be dataframe, model ...)
    """
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print("File saved at ", path)


def load_pickle(path):
    """
    loads in pickled file - use to load in dataframe, model etc.
    :param path: path to load form
    :return: unpickled file
    """
    with open(path, "rb") as f:
        return pickle.load(f)
    print("File loaded: ", path)


def load_app_training_data():
    app_train = pd.read_csv('../input/application_train.csv')
    print('Training data shape: ', app_train.shape)
    return app_train


def load_test_data():
    # Testing data features
    app_test = pd.read_csv('../input/application_test.csv')
    print('Testing data shape: ', app_test.shape)
    return app_test


def encode_binary_cols(train: pd.DataFrame, test: pd.DataFrame):
    """
    Applies label encoding to the train and test dataframes.
    Will only do this if there are more than 2 labels (nan is considered a category)

    :param train: training data
    :param test: test data
    :return: train and test data with label encoded columns
    """
    le = LabelEncoder()
    encoded_cols = []
    for col in train:
        if train[col].dtype == 'object':
            # If 2 or fewer unique categories (a nan will count as a category)
            if len(list(train[col].unique())) <= 2:
                # Train on the training data
                le.fit(train[col])
                # Transform both training and testing data
                train[col] = le.transform(train[col])
                test[col] = le.transform(test[col])
                encoded_cols.append(col)
    print("Label encoded columns", encoded_cols)

    return train, test


def one_hot_encode(train, test):
    """
    Applies one hot encoding to given dataframes
    :param train: training data
    :param test: testing data
    :return: dataframes with one hot encoding applied to them
    """
    # Dummy encoding will not create a column for nans
    train = pd.get_dummies(train)
    test = pd.get_dummies(test)

    print("AFTER ONE HOT ENCODING")
    print('Training Features shape: ', train.shape)
    print('Testing Features shape: ', test.shape)
    return train, test


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


def get_train_labels(train: pd.DataFrame):
    """
    Return the train labels
    :param train: training dataframe containing "TARGET
    :return: training labels i.e. y values
    """
    return train['TARGET']


def remove_days_employed_anomaly(train, test):
    """
    Removes the 365243 anomaly (from DAYS_EMPLOYED) by replacing it with a nan.
    Also creates a binary feature indicating
    at which rows this anomalous value occurred
    :param train: training dataframe containing the DAYS_EMPLOYED columns
    :param test: test dataframe
    :return: train and test dataframes with the anomaly removed
    """
    # Make anomalous flag column
    train['DAYS_EMPLOYED_ANOM'] = train["DAYS_EMPLOYED"] == 365243
    # Replace anomalous values with nan
    train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

    # train['DAYS_EMPLOYED'].plot.hist(title='Days Employment Histogram')
    # plt.xlabel('Days Employment')

    # Repeat with test column
    test['DAYS_EMPLOYED_ANOM'] = test["DAYS_EMPLOYED"] == 365243
    test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    print('Test data contain %d anomalies out of %d rows' % (
        test["DAYS_EMPLOYED_ANOM"].sum(), len(test)))
    return train, test


def remove_missing_cols(train, test, thr=0.68):
    """
    Removes columns from the training data with over a certain threshold of missing values. The test data is then
    aligned
    :param train: dataframe
    :param test: dataframe
    :param thr: If less than the threshold the column will be removed
    :return:
    """
    print("Removing columns with {} proportion of missing values".format(thr))
    train = train.loc[:,
            train.isnull().mean() < thr]  # remove all columns with more than x% missing values
    align_data(train, test, verbose=False)

    print("AFTER REMOVING MISSING COLS (and aligning):")
    print('Training Features shape: ', train.shape)
    print('Testing Features shape: ', test.shape)
    return train, test


# def mean_impute(df):
#     """
#     Applies mean imputation to the dataframe to fill in the nans
#     :param df: dataframe
#     :return: dataframe with means imputed
#     """
#     return df.fillna(df.mean())


def mean_imputation(train: pd.DataFrame, test: pd.DataFrame):
    """
    Applies mean imputation to all columns with missing values in the the train and test data
    The imputer is fitted on the training data only and applied to both train and test, meaning that both dataframes
    require to be aligned beforehand
    :param train: Training data
    :param test: Test data
    :return: Dataframes of the train and test with all columns mean imputed
    """
    imputer = Imputer(strategy='mean')
    # Fit on the training data
    imputer.fit(train)
    # Transform both training and testing data
    train[train.columns] = imputer.transform(train[train.columns])
    test[test.columns] = imputer.transform(test[test.columns])

    print("AFTER MEAN IMPUTATION:")
    print('Training data shape: ', train.shape)
    print('Testing data shape: ', test.shape)

    return train, test


def normalise(train, test):
    """
    Normalises features in the test and train dataframes to be between 0-1
    MAKE SURE SK_CURR_ID AND TARGET have been dropped!
    :param train: training dataframe
    :param test: test dataframe
    :return: normalised train and test dataframes
    """
    assert "TARGET" not in train, "TARGET column should be dropped in train"
    assert "SK_ID_CURR" not in train, "SK_ID_CURR column should be dropped in train"
    assert "SK_ID_CURR" not in test, "SK_ID_CURR column should be dropped in test"

    scaler = MinMaxScaler(feature_range=(0, 1))  # Scale each feature to 0-1
    scaler.fit(train)
    train[train.columns] = scaler.transform(train[train.columns])
    test[test.columns] = scaler.transform(test[test.columns])

    print("AFTER NORMALISATION:")
    print('Training data shape: ', train.shape)
    print('Testing data shape: ', test.shape)
    return train, test


def neg_days_to_years(train: pd.DataFrame, test: pd.DataFrame):
    """
    Converts all days columns to positive values with a years unit. Call this after dealing with DAYS_EMPLOYED anomalies
    NOTE - may want to pass in as copy otherwise the dataframe will be updated
    :param train:
    :param test:
    :return: train and test dataframes with days columns positive and in years
    """

    for col in train:
        if "DAYS" in col and col != 'DAYS_EMPLOYED_ANOM':
            train[[col]] = -train[col] / 365
            test[[col]] = -test[col] / 365

    #             print(col, (train[col].dropna()>=0).all()) # drop na's to show DAYS_LAST_PHONE_CHANGE is valid
    return train, test


def box_cox_transform(df: pd.DataFrame):
    """
    Apply box cox transformations to columns. Only works for numerical colums with positive values only
    Only applied to numerical columns (excluding SK_ID_CURR)
    :param df: Data frame to apply box cox transformations to
    :return: transfomed dataframe and dict mapping the column to the lambda transformation value
    """
    lambda_map = {}
    for col in df:
        if df[col].dtype != 'object' and (df[col].dropna() > 0).all() and col != 'SK_ID_CURR':
            df[[col]], lamda = boxcox(df[col])
            lambda_map[col] = lamda
    return df, lambda_map


def create_and_save_submission(app_test: pd.DataFrame, predictions, save_path: str):
    """
    Create a submission for the test predictions which can be uploaded onto the kaggle submission
    :param app_test: Original app test dataframe -> Must include the column SK_ID_CURR
    :param predictions: array containing predictions for the test data i.e. y hat
    :param save_path: path to save the submission
            - Should end as a .csv
    """
    # Submission dataframe
    submit = app_test[['SK_ID_CURR']]
    submit['TARGET'] = predictions

    # Save the submission to a csv file
    submit.to_csv(save_path, index=False)
    print("Predictions saved to: ", save_path)


def cross_val_roc_curve(train_X: np.array, train_Y: np.array, classifier):
    """
    Creates a ROC AUC based off a 5 split stratified cross validation on the training data.
    Taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    :param train_X: Training data (with no target)
    :param train_Y: Target values
    :param classifier: model to test on e.g. logistic regression
    """

    # ROC AUC with stratified cross validation
    X = train_X
    y = train_Y

    cv = StratifiedKFold(n_splits=5, shuffle=True)
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
    plt.title('ROC AUC')
    plt.legend(loc="lower right")

    print("Avg ROC AUC score: {}".format(np.mean(aucs)))


def feature_aic_bic(train: pd.DataFrame, feature_name: str):
    """
    Calculates and prints the statistical summary from regression results
    (Includes statistical significance, AIC, BIC etc.)
    :param train: Training data
    :param feature_name: name of the column to get the statistical breakdown of
    :return: None
    """
    # calculates the aic and bic values between the target and a column
    # http://www.differencebetween.net/miscellaneous/difference-between-aic-and-bic/

    data = train.copy()
    data = data[[feature_name, "TARGET"]]

    # mean impute for any missing values
    imputer = Imputer(strategy='mean')
    imputer.fit(data)
    imputed_data = imputer.transform(data)
    data[feature_name] = imputed_data

    data["intercept"] = 1.0
    logit = sm.Logit(data["TARGET"], data[feature_name])
    result = logit.fit()

    print("Selected Feature", feature_name)
    result.summary2()  # full summary
    # print("AIC", result.aic)
    # print("BIC", result.bic)


def imputed_col_aic(data: pd.DataFrame, feature_name: str):
    """
    Calculates the AIC score of a imputed column against the TARGET
    :param data: Dataframe containing the 'TARGET' column and the feature to compare against
    :param feature_name: name of the feature to get the aic score of
    :return: AIC score
    """
    logit = sm.Logit(data["TARGET"], data[feature_name])
    result = logit.fit()
    return result.aic


def oversample(X: pd.DataFrame, y: pd.DataFrame, technique: str = 'adasyn'):
    """
    Oversamples the minority class to balance the classes
    :param X: unbalanced dataset as a dataframe
    :param y: labels for the dataset
    :param technique: either 'SMOTE' or 'ADASYN'
    :return: the balanced dataset and labels
    """
    if technique is 'adasyn':
        os_method = ADASYN()
    elif technique is 'smote':
        os_method = SMOTE()
    X, y = os_method.fit_sample(X, y)
    return X, y


def plot_feature_importances(fi_df, n=15):
    """
    Plots top n features from the feature importances assigned by a model (i.e. lgbm)
    Taken from https://www.kaggle.com/willkoehrsen/automated-feature-engineering-basics

    Parameters
    --------
        fi_df :
            feature importances. Must have the features in a column
            called `features` and the importances in a column called `importance
        n: int
            number of features to plot

    Return
    -------
        shows a plot of the 15 most importance features

        df : dataframe
            feature importances sorted by importance (highest to lowest)
            with a column for normalized importance
        """

    # Sort features according to importance
    fi_df = fi_df.sort_values('importance', ascending=False).reset_index()

    # Normalise importances
    fi_df['importance_normalized'] = fi_df['importance'] / fi_df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(14, 10))
    ax = plt.subplot()

    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(fi_df.index[:n]))),
            fi_df['importance_normalized'].head(n),
            align='center', edgecolor='k')

    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(fi_df.index[:n]))))
    ax.set_yticklabels(fi_df['feature'].head(n))

    # Plot labeling
    plt.xlabel('Normalised Importance')
    plt.title('Feature Importances')
    plt.show()

    return fi_df

def lgbm_objective(hyperparameters, train_set, iteration, nfold=5):
    """
    Objective function for grid and random search. Returns
    the cross validation score from a set of hyperparameters.
    :param hyperparameters: hyperparameters instance
    :param train_set: an LGBM dataset containing the training data and labels
    :param iteration: the CV iteration number these hyperparameters were in
    :param nfold: number of CV folds
    :return : an array containing the score achieved by the current hyperparameters instance, the instance itself and the iteration number
    """
    
    # Number of estimators will be found using early stopping
    if 'n_estimators' in hyperparameters.keys():
        del hyperparameters['n_estimators']
    
     # Perform n_folds cross validation
    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round = 2000, nfold = nfold, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = 1001)
    
    # results to retun
    score = cv_results['auc-mean'][-1]
    estimators = len(cv_results['auc-mean'])
    hyperparameters['n_estimators'] = estimators 
    
    return [score, hyperparameters, iteration]

def lgbm_hyperopt_objective(hyperparameters, train_set, nfold=5, out_file=None):
    """
    Objective function for Gradient Boosting Machine Hyperparameter Optimization.
    Writes a new line to `outfile` on every iteration
    :param hyperparameters: hyperparameters instance
    :param train_set: an LGBM dataset containing the training data and labels
    :param iteration: the CV iteration number these hyperparameters were in
    :param nfold: number of CV folds
    :return : a dictionary with information useful for evaluating the performance of a specific hyperparameters selection
    """
    
    # Keep track of global variable
    global  ITERATION

    ITERATION += 1
    
    # Using early stopping to find number of trees trained
    if 'n_estimators' in hyperparameters:
        del hyperparameters['n_estimators']
    
    # Retrieve the subsample
    subsample = hyperparameters['boosting_type'].get('subsample', 1.0)
    
    # Extract the boosting type and subsample to top level keys
    hyperparameters['boosting_type'] = hyperparameters['boosting_type']['boosting_type']
    hyperparameters['subsample'] = subsample
    
    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])

    start = timer()
    
    # Perform n_folds cross validation
    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round = 10000, nfold = nfold, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = 50, verbose_eval=False)

    run_time = timer() - start
    
    # Extract the best score
    best_score = cv_results['auc-mean'][-1]
    
    # Loss must be minimized
    loss = 1 - best_score
    
    # Boosting rounds that returned the highest cv score
    n_estimators = len(cv_results['auc-mean'])
    
    # Add the number of estimators to the hyperparameters
    hyperparameters['n_estimators'] = n_estimators

    if out_file:
        # Write to the csv file ('a' means append)
        of_connection = open(out_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow([loss, hyperparameters, ITERATION, run_time, best_score])
        of_connection.close()

    # Dictionary with information for evaluation
    return {'loss': loss, 'hyperparameters': hyperparameters, 'iteration': ITERATION,
            'train_time': run_time, 'status': STATUS_OK}

def random_search(trainX, trainY, nfold = 5, out_file = None, max_evals = 100):
    """
    Random search for hyperparameter optimization
    :param train_X: Training data (no labels)
    :param train_Y: Training labels
    :param param_grid: Grid of hyperparameters
    :param out_file: A filename to which info about parameter tuning is appended
    :param max_evals: Upper bound for number of evaluations
    :return results: A mapping between each cross-validation iteration and the resulting hyperparameters
    """

    # Create LGB dataset from trainX, trainY
    train_set = lgb.Dataset(data = trainX, label = trainY)
    
    # Hyperparameter grid
    param_grid = {
        'boosting_type': ['gbdt', 'goss', 'dart'],
        'num_leaves': list(range(20, 150)),
        'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),
        'subsample_for_bin': list(range(20000, 300000, 20000)),
        'min_child_samples': list(range(20, 500, 5)),
        'reg_alpha': list(np.linspace(0, 1)),
        'reg_lambda': list(np.linspace(0, 1)),
        'colsample_bytree': list(np.linspace(0.6, 1, 10)),
        'subsample': list(np.linspace(0.5, 1, 100)),
        'is_unbalance': [True, False]
    }
    
    # Dataframe for results
    results = pd.DataFrame(columns = ['score', 'params', 'iteration'],
                                  index = list(range(max_evals)))
    
    # Keep searching until reach max evaluations
    for i in tqdm(range(max_evals)):
        
        # Choose random hyperparameters
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        hyperparameters['subsample'] = 1.0 if hyperparameters['boosting_type'] == 'goss' else hyperparameters['subsample']

        # Evaluate randomly selected hyperparameters
        eval_results = lgbm_objective(hyperparameters, train_set, i, nfold)
        
        results.loc[i, :] = eval_results
        
        if out_file:
            # open connection (append option) and write results
            of_connection = open(out_file, 'a')
            writer = csv.writer(of_connection)
            writer.writerow(eval_results)

            # make sure to close connection
            of_connection.close()
            
    
    # Sort with best score on top
    results.sort_values('score', ascending = False, inplace = True)
    results.reset_index(inplace = True)
    return results


def hyperoptTPE(trainX, trainY, nfold = 5, out_file = None, max_evals = 100):
    """
    Random search for hyperparameter optimization
    :param train_X: Training data (no labels)
    :param train_Y: Training labels
    :param param_grid: Grid of hyperparameters
    :param out_file: A filename to which info about parameter tuning is appended
    :param max_evals: Upper bound for number of evaluations
    :return trials_dict: A dictionary of Hyperopt trials and the corresponding hyperparameters found for each
    """
    
    # Create LGB dataset from trainX, trainY
    train_set = lgb.Dataset(data = trainX, label = trainY)
    
    # Define the search space
    space = {
        'boosting_type': hp.choice('boosting_type', 
                                                [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
#                                                  {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                                 {'boosting_type': 'goss', 'subsample': 1.0}]),
        'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)),
        'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
        'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
        'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
        'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
        'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
        'is_unbalance': hp.choice('is_unbalance', [True, False]),
    }
    
    # Create the algorithm
    tpe_algorithm = tpe.suggest
    
    if out_file:
        # Create a file and open a connection
        of_connection = open(out_file, 'w')
        writer = csv.writer(of_connection)
        # Write column names
        headers = ['loss', 'hyperparameters', 'iteration', 'runtime', 'score']
        writer.writerow(headers)
        of_connection.close()

    # Global variable
    global  ITERATION

    ITERATION = 0
    
    # Record results
    trials = Trials()
    
    # Prepare objective function to be just a function of one parameter: hyperparameters
    objective = partial(lgbm_hyperopt_objective, train_set=train_set, nfold=nfold, out_file=out_file)
    
    # Run optimization
    best = fmin(fn = objective, space = space, algo = tpe.suggest, trials = trials,
            max_evals = max_evals)
    
    # Sort the trials with lowest loss (highest AUC) first
    trials_dict = pd.DataFrame(sorted(trials.results, key = lambda x: x['loss']))

    return trials_dict