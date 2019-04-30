import modules.utils as utils
import os
import gc
import modules.preprocessing as pp
import modules.manual_engineering as me

import pandas as pd


def drop_cols(df, app):
    for column in df.columns.values.tolist():
        if column != 'SK_ID_CURR':
            try:
                app.drop(column, axis=1)
            except:
                print("Could not drop column", column)

    del df
    return app


def make_manual_app():
    cur_dir = os.getcwd()
    app_train = utils.load_pickle(cur_dir + '\\..\\pre_processed_data\\app_train_processed')
    app = me.convert_types(app_train)

    gc.enable()
    del app_train
    gc.collect()

    app['LOAN_RATE'] = app['AMT_ANNUITY'] / app['AMT_CREDIT']
    app['CREDIT_INCOME_RATIO'] = app['AMT_CREDIT'] / app['AMT_INCOME_TOTAL']
    app['EMPLOYED_BIRTH_RATIO'] = app['DAYS_EMPLOYED'] / app['DAYS_BIRTH']
    app['EXT_SOURCE_SUM'] = app[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis=1)
    app['AMT_REQ_SUM'] = app[[x for x in app.columns if 'AMT_REQ_' in x]].sum(axis=1)
    app['CREDIT_TO_GOODS_RATIO'] = app['AMT_CREDIT'] / app['AMT_GOODS_PRICE']
    app['INCOME_PER_CHILD'] = app['AMT_INCOME_TOTAL'] / (1 + app['CNT_CHILDREN'])
    app['PAYMENT_RATE'] = app['AMT_ANNUITY'] / app['AMT_CREDIT']
    app['INCOME_PER_PERSON'] = app['AMT_INCOME_TOTAL'] / app['CNT_FAM_MEMBERS']

    gc.enable()
    bureau = utils.load_pickle(cur_dir + '\\..\\pre_processed_data\\bureau_processed')
    bureau = me.convert_types(bureau)
    # bureau['LOAN_RATE'] = bureau['AMT_ANNUITY'] / bureau['AMT_CREDIT_SUM']
    # bureau_info = bureau.copy()
    bureau_info = me.agg_numeric(bureau, 'SK_ID_CURR', 'BUREAU')

    bureau_balance = utils.load_pickle(cur_dir + '\\..\\pre_processed_data\\bureau_balance_processed')
    bureau_balance = me.convert_types(bureau_balance)

    bureau_balance_info = me.agg_grandchild(bureau_balance, bureau, 'SK_ID_BUREAU', 'SK_ID_CURR', 'BB')
    print('done')
    del bureau_balance, bureau

    app = app.set_index('SK_ID_CURR')
    app = app.merge(bureau_info, on='SK_ID_CURR', how='left')
    del bureau_info

    app = app.merge(bureau_balance_info, on='SK_ID_CURR', how='left')
    del bureau_balance_info

    gc.collect()
    previous = utils.load_pickle(cur_dir + '\\..\\pre_processed_data\\previous_processed')
    previous = me.convert_types(previous)

    previous['LOAN_RATE'] = previous['AMT_ANNUITY'] / previous['AMT_CREDIT']
    previous["AMT_DIFFERENCE"] = previous['AMT_CREDIT'] - previous['AMT_APPLICATION']

    previous_info = me.agg_numeric(previous, 'SK_ID_CURR', 'PREVIOUS')
    # previous_info = previous.copy()
    app = app.merge(previous_info, on='SK_ID_CURR', how='left')
    del previous_info

    installments = utils.load_pickle(cur_dir + '\\..\\pre_processed_data\\installments_processed')
    installments = me.convert_types(installments)

    installments['LATE'] = installments['DAYS_ENTRY_PAYMENT'] > installments['DAYS_INSTALMENT']
    installments['LOW_PAYMENT'] = installments['AMT_PAYMENT'] < installments['AMT_INSTALMENT']
    installments_info = me.agg_grandchild(installments, previous, 'SK_ID_PREV', 'SK_ID_CURR', 'IN')
    del installments

    app = app.merge(installments_info, on='SK_ID_CURR', how='left')
    del installments_info

    cash = utils.load_pickle(cur_dir + '\\..\\pre_processed_data\\cash_processed')
    cash = me.convert_types(cash)

    cash['LATE_PAYMENT'] = cash['SK_DPD'] > 0.0
    cash['INSTALLMENTS_PAID'] = cash['CNT_INSTALMENT'] - cash['CNT_INSTALMENT_FUTURE']
    cash_info = me.agg_grandchild(cash, previous, 'SK_ID_PREV', 'SK_ID_CURR', 'CASH')
    del cash
    app = app.merge(cash_info, on='SK_ID_CURR', how='left')
    del cash_info

    credit = utils.load_pickle(cur_dir + '\\..\\pre_processed_data\\credit_card_processed')
    credit = me.convert_types(credit)

    credit['OVER_LIMIT'] = credit['AMT_BALANCE'] > credit['AMT_CREDIT_LIMIT_ACTUAL']
    credit['BALANCE_CLEARED'] = credit['AMT_BALANCE'] == 0.0
    credit['LOW_PAYMENT'] = credit['AMT_PAYMENT_CURRENT'] < credit['AMT_INST_MIN_REGULARITY']
    credit['LATE'] = credit['SK_DPD'] > 0.0
    credit_info = me.agg_grandchild(credit, previous, 'SK_ID_PREV', 'SK_ID_CURR', 'CC')
    del credit, previous
    app = app.merge(credit_info, on='SK_ID_CURR', how='left')
    del credit_info

    gc.enable()
    gc.collect()

    app['NO_INQUIRIES_MON_TO_BIRTH'] = app['AMT_REQ_CREDIT_BUREAU_MON'] / app['DAYS_BIRTH']
    app['NO_INQUIRIES_DAY_TO_BIRTH'] = app['AMT_REQ_CREDIT_BUREAU_DAY'] / app['DAYS_BIRTH']
    app['NO_INQUIRIES_WEEK_TO_BIRTH'] = app['AMT_REQ_CREDIT_BUREAU_WEEK'] / app['DAYS_BIRTH']
    app['avg_external_source'] = (app['EXT_SOURCE_1'] + app['EXT_SOURCE_2'] + app['EXT_SOURCE_3']) / 3
    app['INCOME_AVG_EXTERNAL_SOURCE'] = app['AMT_INCOME_TOTAL'] / app['avg_external_source']
    app['SOCIAL_CIRCLE_OBS_30_TO_INCOME'] = app['OBS_30_CNT_SOCIAL_CIRCLE'] / app['AMT_INCOME_TOTAL']
    app['SOCIAL_CIRCLE_DEF_30_TO_INCOME'] = app['DEF_30_CNT_SOCIAL_CIRCLE'] / app['AMT_INCOME_TOTAL']
    app['SOCIAL_CIRCLE_OBS_60_TO_INCOME'] = app['OBS_60_CNT_SOCIAL_CIRCLE'] / app['AMT_INCOME_TOTAL']
    app['SOCIAL_CIRCLE_DEF_60_TO_INCOME'] = app['OBS_60_CNT_SOCIAL_CIRCLE'] / app['AMT_INCOME_TOTAL']

    print('After manual feature engineering, there are {} features.'.format(app.shape[1] - 2))

    gc.collect()
    # utils.save_pickle('manual_engineered_features', app)
    app.to_csv('manual_data_added.csv')

    # corrs = app.corr()
    # corrs = corrs.sort_values('TARGET', ascending=False)
    # cor = pd.DataFrame(corrs['TARGET'])
    # print(pd.DataFrame(corrs['TARGET'].head(30)))
    # utils.save_pickle("correlations_manual", pd.DataFrame(cor))
    # cor.to_csv('correlations.csv')


# pp.make_processed_data()  # only needs to be called once
make_manual_app()


