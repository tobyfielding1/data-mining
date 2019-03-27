from scipy.stats import uniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc

from utils import *


def baseline_log_reg(train, train_Y, test, save_path, C=0.0001):
    # Make the model with the specified regularization parameter
    model = LogisticRegression(C=C)

    # Train on the training data
    model.fit(train, train_Y)

    # Make predictions - only require 2nd columns (representing the probability that the target is 1)
    predictions = model.predict_proba(test)[:, 1]

    if save_path is not None:
        # Save model
        save_pickle(save_path, model)  # save model
        print("Log reg baseline model saved to: ", save_path)

    return model, predictions


def random_search_log_reg(train, train_Y, test, save_path):
    model = LogisticRegression()

    # Search parameters and search space
    C = uniform(loc=0, scale=4)
    hyperparameters = dict(C=C)

    # Create randomized search 5-fold cross validation and 100 iterations
    clf = RandomizedSearchCV(model, hyperparameters, random_state=1, n_iter=10, cv=3, verbose=3,
                             n_jobs=10)  # will take a while to run
    # Fit randomized search
    best_model = clf.fit(train, train_Y)

    print('Best C:', best_model.best_estimator_.get_params()['C'])

    predictions = best_model.predict_proba(test)[:, 1]

    if save_path is not None:
        # Save model
        save_pickle(save_path, best_model)  # save model
        print("Log reg baseline model saved to: ", save_path)

    return best_model, predictions


def gbm_basic(train_X: np.array, train_Y, test_X, feature_names, model_save_path, n_folds=5):
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

        # Save model - will keep overwriting so saves last model in the end
        save_pickle(model_save_path, model)

        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
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

    return test_predictions, feature_importances, metrics
