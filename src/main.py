import modules.utils as utils
import os
import modules.preprocessing as pp
import modules.manual_engineering as me

import pandas as pd


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

    if not os.path.isdir(dir + '\\..\\pre_processed_data'):
        os.makedirs(dir + '\\..\\pre_processed_data')

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


def make_manual_app():
    cur_dir = os.getcwd()
    app_train = utils.load_pickle(cur_dir + '\\..\\pre_processed_data\\app_train_processed')
    bureau = utils.load_pickle(cur_dir + '\\..\\pre_processed_data\\bureau_processed')
    previous = utils.load_pickle(cur_dir + '\\..\\pre_processed_data\\previous_processed')
    bureau_balance = utils.load_pickle(cur_dir + '\\..\\pre_processed_data\\bureau_balance_processed')
    cash = utils.load_pickle(cur_dir + '\\..\\pre_processed_data\\cash_processed')
    credit = utils.load_pickle(cur_dir + '\\..\\pre_processed_data\\credit_card_processed')
    installments = utils.load_pickle(cur_dir + '\\..\\pre_processed_data\\installments_processed')

    app = app_train.copy()

    app = me.convert_types(app)
    bureau = me.convert_types(bureau)
    bureau_balance = me.convert_types(bureau_balance)
    previous = me.convert_types(previous)
    cash = me.convert_types(cash)
    credit = me.convert_types(credit)
    installments = me.convert_types(installments)

    app['LOAN_RATE'] = app['AMT_ANNUITY'] / app['AMT_CREDIT']
    app['CREDIT_INCOME_RATIO'] = app['AMT_CREDIT'] / app['AMT_INCOME_TOTAL']
    app['EMPLOYED_BIRTH_RATIO'] = app['DAYS_EMPLOYED'] / app['DAYS_BIRTH']
    app['EXT_SOURCE_SUM'] = app[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis=1)
    app['EXT_SOURCE_MEAN'] = app[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    app['AMT_REQ_SUM'] = app[[x for x in app.columns if 'AMT_REQ_' in x]].sum(axis=1)
    app['CREDIT_TO_GOODS_RATIO'] = app['AMT_CREDIT'] / app['AMT_GOODS_PRICE']
    app['income_per_child'] = app['AMT_INCOME_TOTAL'] / (1 + app['CNT_CHILDREN'])
    app['payment_rate'] = app['AMT_ANNUITY'] / app['AMT_CREDIT']
    app['income_per_person'] = app['AMT_INCOME_TOTAL'] / app['CNT_FAM_MEMBERS']
    app['payment_rate'] = app['AMT_ANNUITY'] / app['AMT_CREDIT']

    utils.save_pickle("app_manual", app)

    # bureau['LOAN_RATE'] = bureau['AMT_ANNUITY'] / bureau['AMT_CREDIT_SUM']
    # bureau_info = me.agg_numeric(bureau, 'SK_ID_CURR', 'BUREAU')

    # bureau_balance['PAST_DUE'] = bureau_balance['STATUS'].isin(['1', '2', '3', '4', '5'])
    # bureau_balance['ON_TIME'] = bureau_balance['STATUS'] == '0'
    bureau_balance_info = me.agg_grandchild(bureau_balance, bureau, 'SK_ID_BUREAU', 'SK_ID_CURR', 'BB')
    print('done')
    del bureau_balance

    app = app.set_index('SK_ID_CURR')
    app = app.merge(bureau, on='SK_ID_CURR', how='left')
    del bureau

    app = app.merge(bureau_balance_info, on='SK_ID_CURR', how='left')
    del bureau_balance_info
    utils.save_pickle("app_join_bureau_manual", app)

    previous['LOAN_RATE'] = previous['AMT_ANNUITY'] / previous['AMT_CREDIT']
    previous["AMT_DIFFERENCE"] = previous['AMT_CREDIT'] - previous['AMT_APPLICATION']
    # previous_info = me.agg_numeric(previous, 'SK_ID_CURR', 'PREVIOUS')
    app = app.merge(previous, on='SK_ID_CURR', how='left')
    del previous
    utils.save_pickle("app_join_bureau_previous_manual", app)

    installments['LATE'] = installments['DAYS_ENTRY_PAYMENT'] > installments['DAYS_INSTALMENT']
    installments['LOW_PAYMENT'] = installments['AMT_PAYMENT'] < installments['AMT_INSTALMENT']
    installments_info = me.agg_grandchild(installments, previous, 'SK_ID_PREV', 'SK_ID_CURR', 'IN')
    del installments

    app = app.merge(installments_info, on='SK_ID_CURR', how='left')
    del installments_info
    utils.save_pickle("app_join_bureau_previous_installments_manual", app)

    cash['LATE_PAYMENT'] = cash['SK_DPD'] > 0.0
    cash['INSTALLMENTS_PAID'] = cash['CNT_INSTALMENT'] - cash['CNT_INSTALMENT_FUTURE']
    cash_info = me.agg_grandchild(cash, previous, 'SK_ID_PREV', 'SK_ID_CURR', 'CASH')
    del cash
    app = app.merge(cash_info, on='SK_ID_CURR', how='left')
    del cash_info
    utils.save_pickle("app_join_bureau_previous_installments_cash_manual", app)

    credit['OVER_LIMIT'] = credit['AMT_BALANCE'] > credit['AMT_CREDIT_LIMIT_ACTUAL']
    credit['BALANCE_CLEARED'] = credit['AMT_BALANCE'] == 0.0
    credit['LOW_PAYMENT'] = credit['AMT_PAYMENT_CURRENT'] < credit['AMT_INST_MIN_REGULARITY']
    credit['LATE'] = credit['SK_DPD'] > 0.0
    credit_info = me.agg_grandchild(credit, previous, 'SK_ID_PREV', 'SK_ID_CURR', 'CC')
    del credit, previous
    app = app.merge(credit_info, on='SK_ID_CURR', how='left')
    del credit_info
    utils.save_pickle("app_join_all", app)

    # app['amt_balance_to_income'] = app['CREDIT_BALANCE_AMT_BALANCE'] / app['AMT_INCOME_TOTAL']
    app['no_inquiries_MON_to_birth'] = app['AMT_REQ_CREDIT_BUREAU_MON'] / app['DAYS_BIRTH']
    app['no_inquiries_DAY_to_birth'] = app['AMT_REQ_CREDIT_BUREAU_DAY'] / app['DAYS_BIRTH']
    app['no_inquiries_WEEK_to_birth'] = app['AMT_REQ_CREDIT_BUREAU_WEEK'] / app['DAYS_BIRTH']
    app['avg_external_source'] = (app['EXT_SOURCE_1'] + app['EXT_SOURCE_2'] + app['EXT_SOURCE_3']) / 3
    app['income_avg_external_source'] = app['AMT_INCOME_TOTAL'] / app['avg_external_source']
    app['social_circle_obs_30_to_income'] = app['OBS_30_CNT_SOCIAL_CIRCLE'] / app['AMT_INCOME_TOTAL']
    app['social_circle_def_30_to_income'] = app['DEF_30_CNT_SOCIAL_CIRCLE'] / app['AMT_INCOME_TOTAL']
    app['social_circle_obs_60_to_income'] = app['OBS_60_CNT_SOCIAL_CIRCLE'] / app['AMT_INCOME_TOTAL']
    app['social_circle_def_60_to_income'] = app['OBS_60_CNT_SOCIAL_CIRCLE'] / app['AMT_INCOME_TOTAL']

    utils.save_pickle("app_join_all_manual", app)

    print('After manual feature engineering, there are {} features.'.format(app.shape[1] - 2))

    targ = dict()
    targ['TARGET'] = app['TARGET']
    cor = targ.corrwith(targ, axis=1)
    # corrs = app.corr()
    # corrs = corrs.sort_values('TARGET', ascending=False)
    print(pd.DataFrame(cor.head(30)))
    utils.save_pickle("manual_engineered_data", app)


# make_processed_data()  # only needs to be called once
make_manual_app()
# call function to create manual features
# find the correlation for each of these new features
# add the features that have a higher correlation to the df
# save the df as a pickle


