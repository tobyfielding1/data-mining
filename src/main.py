import utils
import os
import preprocessing as pp
import manual_engineering as me


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
    utils.save_pickle(dir + '\\..\\pre_processed_data\\credit_card_balance_processed', credit_card_balance)

    installments_payments, orig = pre_process(dir + '\\..\\..\\data\\installments_payments.csv', ['SK_ID_PREV', 'SK_ID_CURR'])
    utils.save_pickle(dir + '\\..\\pre_processed_data\\installments_payments_processed', installments_payments)

    POS_CASH_balance, orig = pre_process(dir + '\\..\\..\\data\\POS_CASH_balance.csv', ['SK_ID_PREV', 'SK_ID_CURR'])
    utils.save_pickle(dir + '\\..\\pre_processed_data\\POS_CASH_balance_processed', POS_CASH_balance)

    previous_application, orig = pre_process(dir + '\\..\\..\\data\\previous_application.csv', ['SK_ID_PREV', 'SK_ID_CURR'])
    utils.save_pickle(dir + '\\..\\pre_processed_data\\previous_application_processed', previous_application)


make_processed_data()




