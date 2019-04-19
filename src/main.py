import utils
import os
import preprocessing as pp
import manual_engineering as me


# Calls all of the pre-processing methods on the df given using the pre-processing module
def pre_process(filename, drop_columns):
    orig_data = pp.load_data(filename)
    data = orig_data.copy()
    data.drop(columns=drop_columns, inplace=True)
    data = pp.encode_binary_cols(data)
    data = pp.one_hot_encoding(data)
    data = pp.remove_missing_cols(data)
    data = pp.normalise(data)

    for col in drop_columns:
        print(col)
        data[col] = orig_data[col]

    return data, orig_data


# Calls the pre-processing function for each of the different files and saves them as a pickle file
# ONLY NEEDS TO BE CALLED ONCE
def make_processed_data():
    dir = os.getcwd()
    app_train, orig_train = pre_process(dir + '\\..\\..\\data\\application_train.csv', ['TARGET', 'SK_ID_CURR'])
    app_test, orig_test = pre_process(dir + '\\..\\..\\data\\application_test.csv', ['SK_ID_CURR'])
    app_train, app_test = pp.align_data(app_train, app_test)

    app_train.drop(columns=['TARGET', 'SK_ID_CURR'], inplace=True)
    app_test.drop(columns=['SK_ID_CURR'], inplace=True)

    app_train = pp.mean_imputation(app_train, app_train)  # may be a bit slow
    app_test = pp.mean_imputation(app_test, app_train)  # may be a bit slow

    app_train['TARGET'] = orig_train['TARGET']
    app_train['SK_ID_CURR'] = orig_train['SK_ID_CURR']
    app_test['SK_ID_CURR'] = orig_test['SK_ID_CURR']

    utils.save_pickle(dir + '\\..\\pre_processed_data\\app_train_processed', app_train)
    utils.save_pickle(dir + '\\..\\pre_processed_data\\app_test_processed', app_test)

    bureau, orig = pre_process(dir + '\\..\\..\\data\\bureau.csv', ['SK_ID_BUREAU', 'SK_ID_CURR'])
    utils.save_pickle(dir + '\\..\\pre_processed_data\\bureau_processed', bureau)

    bureau_balance, orig = pre_process(dir + '\\..\\..\\data\\bureau_balance.csv', ['SK_ID_BUREAU'])
    utils.save_pickle(dir + '\\..\\pre_processed_data\\bureau_balance_processed', bureau_balance)

    credit_card_balance, orig = pre_process(dir + '\\..\\..\\data\\credit_card_balance.csv', ['SK_ID_PREV', 'SK_ID_CURR'])
    utils.save_pickle(dir + '\\..\\pre_processed_data\\credit_card_processed', credit_card_balance)

    installments_payments, orig = pre_process(dir + '\\..\\..\\data\\installments_payments.csv', ['SK_ID_PREV', 'SK_ID_CURR'])
    utils.save_pickle(dir + '\\..\\pre_processed_data\\installments_processed', installments_payments)

    POS_CASH_balance, orig = pre_process(dir + '\\..\\..\\data\\POS_CASH_balance.csv', ['SK_ID_PREV', 'SK_ID_CURR'])
    utils.save_pickle(dir + '\\..\\pre_processed_data\\cash_processed', POS_CASH_balance)

    previous_application, orig = pre_process(dir + '\\..\\..\\data\\previous_application.csv', ['SK_ID_PREV', 'SK_ID_CURR'])
    utils.save_pickle(dir + '\\..\\pre_processed_data\\previous_processed', previous_application)


def make_manual_features():
    app = utils.load_pickle(dir + '\\..\\pre_processed_data\\app_train_processed')
    bureau = utils.load_pickle(dir + '\\..\\pre_processed_data\\bureau_processed')
    previous = utils.load_pickle(dir + '\\..\\pre_processed_data\\previous_processed')
    bureau_balance = utils.load_pickle(dir + '\\..\\pre_processed_data\\bureau_balance_processed')
    cash = utils.load_pickle(dir + '\\..\\pre_processed_data\\cash_processed')
    credit = utils.load_pickle(dir + '\\..\\pre_processed_data\\credit_card_processed')
    installments = utils.load_pickle(dir + '\\..\\pre_processed_data\\installments_processed')

    app['LOAN_RATE'] = app['AMT_ANNUITY'] / app['AMT_CREDIT']
    app['CREDIT_INCOME_RATIO'] = app['AMT_CREDIT'] / app['AMT_INCOME_TOTAL']
    app['EMPLOYED_BIRTH_RATIO'] = app['DAYS_EMPLOYED'] / app['DAYS_BIRTH']
    app['EXT_SOURCE_SUM'] = app[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis=1)
    app['EXT_SOURCE_MEAN'] = app[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    app['AMT_REQ_SUM'] = app[[x for x in app.columns if 'AMT_REQ_' in x]].sum(axis=1)
    bureau['LOAN_RATE'] = bureau['AMT_ANNUITY'] / bureau['AMT_CREDIT_SUM']

    bureau_info = me.agg_child(bureau, 'SK_ID_CURR', 'BUREAU')

    bureau_balance['PAST_DUE'] = bureau_balance['STATUS'].isin(['1', '2', '3', '4', '5'])
    bureau_balance['ON_TIME'] = bureau_balance['STATUS'] == '0'
    bureau_balance_info = me.agg_grandchild(bureau_balance, bureau, 'SK_ID_BUREAU', 'SK_ID_CURR', 'BB')
    del bureau_balance, bureau

    app = app.set_index('SK_ID_CURR')
    app = app.merge(bureau_info, on='SK_ID_CURR', how='left')
    del bureau_info

    app = app.merge(bureau_balance_info, on='SK_ID_CURR', how='left')
    del bureau_balance_info

    previous['LOAN_RATE'] = previous['AMT_ANNUITY'] / previous['AMT_CREDIT']
    previous["AMT_DIFFERENCE"] = previous['AMT_CREDIT'] - previous['AMT_APPLICATION']
    previous_info = me.agg_child(previous, 'SK_ID_CURR', 'PREVIOUS')
    app = app.merge(previous_info, on='SK_ID_CURR', how='left')
    del previous_info

    installments['LATE'] = installments['DAYS_ENTRY_PAYMENT'] > installments['DAYS_INSTALMENT']
    installments['LOW_PAYMENT'] = installments['AMT_PAYMENT'] < installments['AMT_INSTALMENT']
    installments_info = me.agg_grandchild(installments, previous, 'SK_ID_PREV', 'SK_ID_CURR', 'IN')
    del installments
    app = app.merge(installments_info, on='SK_ID_CURR', how='left')
    del installments_info

    cash['LATE_PAYMENT'] = cash['SK_DPD'] > 0.0
    cash['INSTALLMENTS_PAID'] = cash['CNT_INSTALMENT'] - cash['CNT_INSTALMENT_FUTURE']
    cash_info = me.agg_grandchild(cash, previous, 'SK_ID_PREV', 'SK_ID_CURR', 'CASH')
    del cash
    app = app.merge(cash_info, on='SK_ID_CURR', how='left')
    del cash_info

    credit['OVER_LIMIT'] = credit['AMT_BALANCE'] > credit['AMT_CREDIT_LIMIT_ACTUAL']
    credit['BALANCE_CLEARED'] = credit['AMT_BALANCE'] == 0.0
    credit['LOW_PAYMENT'] = credit['AMT_PAYMENT_CURRENT'] < credit['AMT_INST_MIN_REGULARITY']
    credit['LATE'] = credit['SK_DPD'] > 0.0
    credit_info = me.agg_grandchild(credit, previous, 'SK_ID_PREV', 'SK_ID_CURR', 'CC')
    del credit, previous
    app = app.merge(credit_info, on='SK_ID_CURR', how='left')
    del credit_info

    print('After manual feature engineering, there are {} features.'.format(app.shape[1] - 2))


make_processed_data()  # only needs to be called once

# call function to create manual features
# find the correlation for each of these new features
# add the features that have a higher correlation to the df
# save the df as a pickle


