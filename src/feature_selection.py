import pandas as pd
import numpy as np
import featuretools as ft
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 22
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import gc


def load_manual_features():
    try:
        print('Attempting to load a cached copy of manual features')
        manual_features = pd.read_pickle('./cached_pickles/cached_manual_features.pkl')
    except:
        print('No cached copy found, loading direct from CSV, this might take a few minutes')
        manual_features = pd.read_csv('./manual_data_added.csv')
        manual_features.to_pickle('./cached_pickles/cached_manual_features.pkl')

    columns = list(manual_features.columns)

    print('Finished loading, %d manually created features discovered\n' % len(columns))
    return manual_features


# def load_data():
#     try:
#         print('Attempting to load a cached copy of training and test data')
#         train = pd.read_pickle('./cached_pickles/cached_application_train.pkl')
#         test = pd.read_pickle('./cached_pickles/cached_application_test.pkl')
#     except:
#         print('No cached copy found, loading pre-processed versions')
#         train = pd.read_pickle('../pre_processed_data/app_train_processed')
#         test = pd.read_pickle('../pre_processed_data/app_test_processed')
#         train.to_pickle('./cached_pickles/cached_application_train.pkl')
#         test.to_pickle('./cached_pickles/cached_application_test.pkl')
#
#     print('Finished loading training and test data\n')
#     return train, test;


def remove_collinear_variables(manual_features):
    try:
        manual_features = pd.read_pickle('./cached_pickles/cached_manual_features_post_collinear.pkl')
        print("Cached copy found, skipping recalculating collinear features\n")
    except:
        threshold = 0.9
        print("Identifying collinear variables, be warned this will take a long time (60+ mins, 10gb RAM)")

        corr_matrix = manual_features.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        print('Unnecessary features removed: %d' % (len(to_drop)))

        manual_features = manual_features.drop(columns=to_drop)

        print('Feature shape after removal: ', manual_features.shape, '\n')
        manual_features.to_pickle('./cached_pickles/cached_manual_features_post_collinear.pkl')

    return manual_features


def remove_missing_values(manual_features):
    try:
        manual_features = pd.read_pickle('./cached_pickles/cached_manual_features_post_missing_values.pkl')
        print('Cached copy found, skipping recalculating missing values\n')
    except:
        print("Identifying features with more than 75% values missing")
        train_missing = (manual_features.isnull().sum() / len(manual_features)).sort_values(ascending=False)
        train_missing.head()

        train_missing = train_missing.index[train_missing > 0.75]

        print('There are %d columns with more than 75%% missing values' % len(train_missing))
        manual_features = pd.get_dummies(manual_features.drop(columns=train_missing))
        print('Feature shape after removals: ', manual_features.shape, '\n')
        manual_features.to_pickle('./cached_pickles/cached_manual_features_post_missing_values.pkl')

    return manual_features


def plot_feature_importance(df, threshold = 0.9):
    plt.rcParams['font.size'] = 18

    # Sort features according to importance
    df = df.sort_values('importance', ascending=False).reset_index()

    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()

    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))),
            df['importance_normalized'].head(15),
            align='center', edgecolor='k')

    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))

    # Plot labeling
    plt.xlabel('Normalized Importance');
    plt.title('Feature Importances')
    plt.show()

    # Cumulative importance plot
    plt.figure(figsize=(8, 6))
    plt.plot(list(range(len(df))), df['cumulative_importance'], 'r-')
    plt.xlabel('Number of Features');
    plt.ylabel('Cumulative Importance');
    plt.title('Cumulative Feature Importance');
    plt.show();

    importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
    print('%d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold))

    return df


def remove_unimportant_features(manual_features):
    try:
        manual_features = pd.read_pickle('./cached_pickles/cached_manual_features_post_unimportant_features.pkl')
        print('Cached copy found, skipping recalculating unimportant features\n')
    except:
        print("Identifying features with no relevance to the solution")

        train_labels = manual_features["TARGET"]
        manual_features = manual_features.drop(columns="TARGET")

        feature_importances = np.zeros(manual_features.shape[1])
        model = lgb.LGBMClassifier(objective='binary', boosting_type='goss', n_estimators=10000,
                                   class_weight='balanced')

        for i in range(2):
            train_features, valid_features, train_y, valid_y = train_test_split(manual_features, train_labels, test_size=0.25,random_state=i)
            model.fit(train_features, train_y, early_stopping_rounds=100, eval_set=[(valid_features, valid_y)],
                      eval_metric='auc', verbose=200)
            feature_importances += model.feature_importances_

        feature_importances = feature_importances / 2
        feature_importances = pd.DataFrame(
            {'feature': list(manual_features.columns), 'importance': feature_importances}).sort_values('importance',
                                                                                             ascending=False)
        zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
        print('There are %d features with 0.0 importance' % len(zero_features))

        norm_feature_importances = plot_feature_importance(feature_importances)

        manual_features = manual_features.drop(columns = zero_features)

        print('Feature shape after removals: ', manual_features.shape, '\n')
        manual_features.to_pickle('./cached_pickles/cached_manual_features_post_unimportant_features.pkl')
    return manual_features


def run_feature_selection():
    print('Starting feature selection on manually created features')
    manual_features = load_manual_features()
    # train, test = load_data()
    manual_features = remove_collinear_variables(manual_features)
    manual_features = remove_missing_values(manual_features)
    manual_features = remove_unimportant_features(manual_features)


run_feature_selection()
