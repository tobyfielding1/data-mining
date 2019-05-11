from scipy.stats import uniform

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import gc
import pprint

from utils import *


def baseline_log_reg(train, train_Y, test, save_path=None, C=0.0001):
    """
    Creates the baseline logistic regression model
    :param train: training data
    :param train_Y: training labels
    :param test: test data
    :param save_path: file path to save model to.
                    - If None will not save the model
    :param C: Tunable C parameter
    :return: Trained model and predictions for the test data (i.e. y hat)
    """
    model = LogisticRegression(C=C)
    model.fit(train, train_Y)

    # Make predictions -> only require 2nd column (representing the probability that the target is 1)
    predictions = model.predict_proba(test)[:, 1]

    if save_path:
        # Save model
        save_pickle(save_path, model)
        print("Log reg baseline model saved to: ", save_path)

    return model, predictions


def random_search_log_reg(train, train_Y, test, save_path=None):
    """
    Uses random search to tune the C parameter for logistic regression and make a prediction.
    :param train: Training data as array
    :param train_Y: Training labels
    :param test: Test data as array
    :param save_path: path to save model to
            - If None then model will not be saved
    :return: best model from last fold and predictions for the test data
    """
    model = LogisticRegression()

    # Search parameters and search space
    C = uniform(loc=0, scale=4)
    hyperparameters = dict(C=C)

    # Create Random search using 3-fold cross validation and 10 iterations
    clf = RandomizedSearchCV(model, hyperparameters, random_state=1, n_iter=5, cv=3, verbose=3,
                             n_jobs=-1)  # will take a while to run
    best_model = clf.fit(train, train_Y)
    print('Best C:', best_model.best_estimator_.get_params()['C'])
    predictions = best_model.predict_proba(test)[:, 1]

    if save_path:
        # Save model
        save_pickle(save_path, best_model)
        print("Log reg baseline model saved to: ", save_path)

    return best_model, predictions

def log_reg_cv(train_X, train_Y, test_X, feature_names, C=0.0001, model_save_path=None, n_folds=5):
    """
    Creates logistic regression models which are cross-validated, averaging ROC AUC scores from every fold.
    :param train_X: Training data as array
    :param train_Y: Training labels
    :param test_X: Test data as array
    :param feature_names: The names of the features
    :param C: Tunable C parameter
    :param model_save_path: Path to save model to
            - If None then model will not be saved
    :param n_folds: number of folds to use
    :return: best model from last fold, predictions for the test data and extracted features importances
    """
    print('Training Data Shape: ', train_X.shape)
    print('Testing Data Shape: ', test_X.shape)

    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=50)

    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))

    # Empty array for test predictions
    test_predictions = np.zeros(test_X.shape[0])

    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(train_X.shape[0])

    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []

    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(train_X):
        # Training data for the fold
        train_features, train_labels = train_X[train_indices], train_Y[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = train_X[valid_indices], train_Y[valid_indices]

        # Create the model
        model = LogisticRegression(C=C)

        # Train the model
        model.fit(train_features, train_labels)

        # Make predictions - get the average from all the folds
        test_predictions += model.predict_proba(test_X)[:, 1] / k_fold.n_splits

        # Record the in and out of fold predictions
        in_fold = model.predict_proba(train_features)[:, 1]
        
        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features)[:, 1]

        # Record the best score
        train_score = roc_auc_score(train_labels, in_fold)
        valid_score = roc_auc_score(valid_labels, out_of_fold[valid_indices])

        valid_scores.append(valid_score)
        train_scores.append(train_score)

        # Save model - will keep overwriting so saves the last fold's model
        if model_save_path:
            save_pickle(model_save_path, model)

        # Clean up memory
        gc.enable()
        # TODO - if becomes very slow potentially due to not gc model (which is returned) 
        del train_features, valid_features
        gc.collect()

    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    
    # Overall validation score
    valid_auc = roc_auc_score(train_Y, out_of_fold)

    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))

    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')

    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores})

    return model, test_predictions, metrics


def random_forest(train_X, train_Y, test_X, feature_names, n_estimators=100, model_save_path=None, n_folds=5):
    """
    Creates a Random Forest model with the specified number of estimators
    :param train_X: Training data as array
    :param train_Y: Training labels
    :param test_X: Test data as array
    :param feature_names: The names of the features
    :param n_estimators: Number of decision trees
    :param model_save_path: Path to save model to
            - If None then model will not be saved
    :param n_folds: number of folds to use
    :return: best model from last fold, predictions for the test data and extracted features importances
    """
    print('Training Data Shape: ', train_X.shape)
    print('Testing Data Shape: ', test_X.shape)

    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=50)

    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))

    # Empty array for test predictions
    test_predictions = np.zeros(test_X.shape[0])

    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(train_X.shape[0])

    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []

    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(train_X):
        # Training data for the fold
        train_features, train_labels = train_X[train_indices], train_Y[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = train_X[valid_indices], train_Y[valid_indices]

        # Create the model
        model = RandomForestClassifier(n_estimators = n_estimators, random_state = 50, verbose = 1, n_jobs = -1)

        # Train the model
        model.fit(train_features, train_labels)

        # Record the feature importances - get the average from all the splits in a fold
        feature_importance_values += model.feature_importances_ / k_fold.n_splits

        # Make predictions - get the average from all the folds
        test_predictions += model.predict_proba(test_X)[:, 1] / k_fold.n_splits

        # Record the in and out of fold predictions
        in_fold = model.predict_proba(train_features)[:, 1]
        
        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features)[:, 1]

        # Record the best score
        train_score = roc_auc_score(train_labels, in_fold)
        valid_score = roc_auc_score(valid_labels, out_of_fold[valid_indices])

        valid_scores.append(valid_score)
        train_scores.append(train_score)

        # Save model - will keep overwriting so saves the last fold's model
        if model_save_path:
            save_pickle(model_save_path, model)

        # Clean up memory
        gc.enable()
        # TODO - if becomes very slow potentially due to not gc model (which is returned) 
        del train_features, valid_features
        gc.collect()

    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    
    # Overall validation score
    valid_auc = roc_auc_score(train_Y, out_of_fold)

    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))

    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')

    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores})

    return model, test_predictions, feature_importances, metrics


def gbm_basic(train_X, train_Y, test_X, feature_names, model_save_path=None, n_folds=5):
    """
    Trains and classifies data using a light gradient boosting model. Train-validation done using k-fold validation
    Taken & modified from https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction

    :param train_X: Training data (no labels)
    :param train_Y: Training labels
    :param test_X: Test data
    :param feature_names: column names (i.e. features)
    :param model_save_path: file path to save classifier model
    :param n_folds: number of folds to use
    :return: (model, test_predictions, feature_importances, metrics)
        - classifier model
        - predictions for the test data
        - dataframe of features and their importances according to the lgbm model
        - train, validation scores for each fold

    """
    print('Training Data Shape: ', train_X.shape)
    print('Testing Data Shape: ', test_X.shape)

    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=50)

    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))

    # Empty array for test predictions
    test_predictions = np.zeros(test_X.shape[0])

    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(train_X.shape[0])

    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []

    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(train_X):
        # Training data for the fold
        train_features, train_labels = train_X[train_indices], train_Y[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = train_X[valid_indices], train_Y[valid_indices]

        # Create the model
        model = lgb.LGBMClassifier(n_estimators=10000, objective='binary',
                                   class_weight='balanced', learning_rate=0.05,
                                   reg_alpha=0.1, reg_lambda=0.1,
                                   subsample=0.8, n_jobs=-1, random_state=50)

        # Train the model
        model.fit(train_features, train_labels, eval_metric='auc',
                  eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names=['valid', 'train'], early_stopping_rounds=100, verbose=200)

        # Record the best iteration
        best_iteration = model.best_iteration_

        # Record the feature importances - get the average from all the splits in a fold
        feature_importance_values += model.feature_importances_ / k_fold.n_splits

        # Make predictions - get the average from all the folds
        test_predictions += model.predict_proba(test_X, num_iteration=best_iteration)[:, 1] / k_fold.n_splits

        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration=best_iteration)[:, 1]

        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']

        valid_scores.append(valid_score)
        train_scores.append(train_score)

        # Save model - will keep overwriting so saves the last fold's model
        if model_save_path:
            save_pickle(model_save_path, model)

        # Clean up memory
        gc.enable()
        # TODO - if becomes very slow potentially due to not gc model (which is returned) 
        del train_features, valid_features
        gc.collect()

    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    # Overall validation score
    valid_auc = roc_auc_score(train_Y, out_of_fold)

    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))

    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')

    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores})

    return model, test_predictions, feature_importances, metrics

def gbm_with_params(train_X, train_Y, test_X, feature_names, hyperparams, model_save_path=None, n_folds=5):
    """
    Trains and classifies data using a light gradient boosting model. Train-validation done using k-fold validation
    It uses a specified dictionary of hyperparameters when training the model.
    
    Taken & modified from https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction

    :param train_X: Training data (no labels)
    :param train_Y: Training labels
    :param test_X: Test data
    :param feature_names: column names (i.e. features)
    :param model_save_path: file path to save classifier model
    :param n_folds: number of folds to use
    :param hyperparams: hyperparameters to be used by the GBM
    :return: (model, test_predictions, feature_importances, metrics)
        - classifier model
        - predictions for the test data
        - dataframe of features and their importances according to the lgbm model
        - train, validation scores for each fold

    """
    print('Training Data Shape: ', train_X.shape)
    print('Testing Data Shape: ', test_X.shape)
    
    # This KFold validation step is only for overfitting purposes
    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=50)

    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))

    # Empty array for test predictions
    test_predictions = np.zeros(test_X.shape[0])

    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(train_X.shape[0])

    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []

    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(train_X):
        # Training data for the fold
        train_features, train_labels = train_X[train_indices], train_Y[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = train_X[valid_indices], train_Y[valid_indices]

        # Create the model
        model = lgb.LGBMClassifier(**hyperparams, random_state = 1001)

        # Train the model
        model.fit(train_features, train_labels, eval_metric='auc',
                  eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names=['valid', 'train'], early_stopping_rounds=100, verbose=200)

        # Record the best iteration
        best_iteration = model.best_iteration_

        # Record the feature importances - get the average from all the splits in a fold
        feature_importance_values += model.feature_importances_ / k_fold.n_splits

        # Make predictions - get the average from all the folds
        test_predictions += model.predict_proba(test_X, num_iteration=best_iteration)[:, 1] / k_fold.n_splits

        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration=best_iteration)[:, 1]

        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']

        valid_scores.append(valid_score)
        train_scores.append(train_score)

        # Save model - will keep overwriting so saves the last fold's model
        if model_save_path:
            save_pickle(model_save_path, model)

        # Clean up memory
        gc.enable()
        # TODO - if becomes very slow potentially due to not gc model (which is returned) 
        del train_features, valid_features
        gc.collect()

    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    # Overall validation score
    valid_auc = roc_auc_score(train_Y, out_of_fold)

    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))

    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')

    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores})

    return model, test_predictions, feature_importances, metrics
    

def gbm_random_search(train_X, train_Y, test_X, feature_names, model_save_path=None, n_folds=5, samples=15000, max_evals=100, out_file=None):
    """
    Trains and classifies data using a light gradient boosting model. Train-validation done using k-fold validation
    This implementation uses random search to find a better hyperparatmeter configuration.
    
    Taken & modified from https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction

    :param train_X: Training data (no labels)
    :param train_Y: Training labels
    :param test_X: Test data
    :param feature_names: column names (i.e. features)
    :param model_save_path: file path to save classifier model
    :param n_folds: number of folds to use
    :param samples: number of data samples to use for hyperparameter tuning (Note that using the whole set is v slow)
    :param out_file: file to write ROC AUCs for multiple random search instances
    :param max_evals: maximum number of model tuning evaluations
    :return: (model, test_predictions, feature_importances, metrics)
        - classifier model
        - predictions for the test data
        - dataframe of features and their importances according to the lgbm model
        - train, validation scores for each fold

    """
    
    # First sample 'sample' no of rows from training data
    trainX_sample = pd.DataFrame(train_X).sample(samples)
    trainY_sample = train_Y[trainX_sample.index]
    
    # Find a good hyperparameter configuration using random search for the selected sample (NOTE: would be too slow to run on the whole thing)
    random_results = random_search(trainX_sample, trainY_sample, nfold=n_folds, out_file=out_file, max_evals=max_evals)
    # Get the best parameters
    random_search_params = random_results.loc[0, 'params']
    
    # Print results from model tuning
    print('The best validation score was {:.5f}'.format(random_results.loc[0, 'score']))
    print('\nThe best hyperparameters were:')
    pprint.pprint(random_search_params)
    
    return gbm_with_params(train_X, train_Y, test_X, feature_names, random_search_params, model_save_path, n_folds=n_folds)


def gbm_bayesian_optim(train_X, train_Y, test_X, feature_names, model_save_path=None, n_folds=5, samples=15000, max_evals=100, out_file=None):
    """
    Trains and classifies data using a light gradient boosting model. Train-validation done using k-fold validation.
    This implementation uses Bayesian Optimization to find a good configuration for the hyperparameters.
    
    Taken & modified from https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction

    :param train_X: Training data (no labels)
    :param train_Y: Training labels
    :param test_X: Test data
    :param feature_names: column names (i.e. features)
    :param model_save_path: file path to save classifier model
    :param n_folds: number of folds to use
    :param samples: number of data samples to use for hyperparameter tuning (Note that using the whole set is v slow)
    :param out_file: file to write ROC AUCs for multiple random search instances
    :param max_evals: maximum number of model tuning evaluations
    :return: (model, test_predictions, feature_importances, metrics)
        - classifier model
        - predictions for the test data
        - dataframe of features and their importances according to the lgbm model
        - train, validation scores for each fold

    """
    
    # First sample 'sample' no of rows from training data
    trainX_sample = pd.DataFrame(train_X).sample(samples)
    trainY_sample = train_Y[trainX_sample.index]
    
    # Find a good hyperparameter configuration using random search for the selected sample (NOTE: would be too slow to run on the whole thing)
    bayesian_results = hyperoptTPE(trainX_sample, trainY_sample, nfold=n_folds, out_file=out_file, max_evals=max_evals)
    # Get the best parameters
    bayesian_params = bayesian_results.loc[0, 'hyperparameters']
    
    # Print results from model tuning
    print('The best validation score was {:.5f}'.format(1-bayesian_results.loc[0, 'loss']))
    print('\nThe best hyperparameters were:')
    pprint.pprint(bayesian_params)
    
    return gbm_with_params(train_X, train_Y, test_X, feature_names, bayesian_params, model_save_path, n_folds=n_folds)

