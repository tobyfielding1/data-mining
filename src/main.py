import utils
import preprocessing as pp


def pre_process(filename):
    data = pp.load_data(filename)
    data = pp.encode_binary_cols(data)
    data = pp.one_hot_encode(data)
    data = pp.remove_missing_cols(data)

    return data


def make_processed_data():
    app_train = pre_process('../data/app_train')
    app_test = pre_process('../data/app_train')
    app_train, app_test = pp.align_data(app_train, app_test)

    utils.save_pickle('pre_processed_data/app_train_processed', app_train)
    utils.save_pickle('pre_processed_data/app_test_processed', app_test)

    app_train = pre_process('../data/app_train')
    utils.save_pickle('pre_processed_data/app_train_processed', app_train)

    app_train = pre_process('../data/app_train')
    utils.save_pickle('pre_processed_data/app_train_processed', app_train)

    app_train = pre_process('../data/app_train')
    utils.save_pickle('pre_processed_data/app_train_processed', app_train)

    app_train = pre_process('../data/app_train')
    utils.save_pickle('pre_processed_data/app_train_processed', app_train)


make_processed_data()




