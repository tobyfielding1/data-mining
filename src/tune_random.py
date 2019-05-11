# imports
import models
import utils

train_data = utils.load_pickle("./final_features_train2.pkl")
test_data = utils.load_pickle("./final_features_test2.pkl")

# get train labels and test id
train_Y = utils.get_train_labels(train_data)
test_ids = test_data[['SK_ID_CURR']]

# drop SK_ID and TARGET
train_data.drop(columns=['TARGET'], inplace=True)
train_data.drop(columns=['SK_ID_CURR'], inplace=True)
test_data.drop(columns=['SK_ID_CURR'], inplace= True)

print(train_data.shape)
print(test_data.shape)

# convert from dataframes to arrays
train_X = train_data.values
test_X = test_data.values
feature_names = train_data.columns # required to create feature importances

model, predictions, feature_importances, metrics = models.gbm_bayesian_optim(train_X, train_Y, test_X, feature_names, samples=50000, max_evals=150)

utils.create_and_save_submission(test_ids, predictions, save_path='../test_predictions/LGBM_bayes_prop.csv')

print(metrics)