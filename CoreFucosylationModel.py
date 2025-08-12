import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from sklearn.svm import OneClassSVM
from sklearn.svm import LinearSVC
# from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MaxAbsScaler
from sklearn.externals import joblib
import DataPreprocess
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score


def MappingConvergence(labeled_x, unlabeled_x, filepath_model):
    # First phase, find strong negative
    positive_x = labeled_x
    positive_y = np.ones(labeled_x.shape[0])

    # use OSVM model from sklearn
    nu_range = np.logspace(-8, -1, 8)
    clf = OneClassSVM(kernel='linear')
    param_grid = dict(nu=nu_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid_search = GridSearchCV(clf, param_grid=param_grid, scoring='accuracy', n_jobs=1, cv=cv, verbose=1)
    grid_search.fit(positive_x, positive_y)
    best_parameters = grid_search.best_estimator_.get_params()
    clf.set_params(nu=best_parameters['nu'])
    clf.fit(labeled_x)

    # outliers from labeled data
    result = clf.predict(labeled_x)
    idx= np.where(result == -1)
    outlier1 = labeled_x[idx[0], 0:]
    print(outlier1.shape[0])
    # outliers from unlabeled data
    result = clf.predict(unlabeled_x)
    idx = np.where(result == -1)
    outlier2 = unlabeled_x[idx[0], 0:]
    print(outlier2.shape[0])
    unlabeled_x = np.delete(unlabeled_x, idx, axis=0)

    new_negative_x = outlier2
    train_x = positive_x
    train_y = positive_y
    iteration_continue = 1
    iteration_n = 0
    iteration_log = []

    # second phase: convergence
    while iteration_continue:
        iteration_n += 1
        train_x = np.concatenate((train_x, new_negative_x), axis=0)
        train_y = np.append(train_y, np.zeros(new_negative_x.shape[0]))
        C_range = np.logspace(-4, 6, 11)
        clf = LinearSVC(class_weight='balanced', dual=False, penalty='l2', max_iter=2000)
        param_grid = dict(C=C_range)
        cv = StratifiedShuffleSplit(n_splits=4, test_size=0.25, random_state=42)
        grid_search = GridSearchCV(clf, param_grid=param_grid, scoring='f1', n_jobs=1, cv=cv)

        grid_search.fit(train_x, train_y)
        best_parameters = grid_search.best_estimator_.get_params()
        clf.set_params(C=best_parameters['C'])
        clf.fit(train_x, train_y)

        result = clf.predict(unlabeled_x)
        idx = np.where(result==0)
        if len(idx[0]):
            iteration_log.append([len(idx[0]), best_parameters['C']])
            new_negative_x = unlabeled_x[idx[0].tolist(), 0:]
            unlabeled_x = np.delete(unlabeled_x, idx[0], axis=0)
        else:
            iteration_continue = 0

    joblib.dump(clf, filepath_model)

    # compute f1 score
    result = clf.predict(train_x)
    tp = 0
    for i in range(len(result)):
        if result[i] and train_y[i]:
            tp += 1
    recall = tp / np.sum(train_y)
    precision = tp / np.sum(result)
    print(2 * (recall * precision) / (recall + precision))

    for num in iteration_log:
        print(num)
        print('\n')
    print('迭代次数:'+str(iteration_n))


def AutoEncoder(train_x, n_encoder1, n_encoder2, n_decoder2, n_decoder1, n_latent=2):
    reg = MLPRegressor(hidden_layer_sizes=(n_encoder1, n_encoder2, n_latent, n_decoder2, n_decoder1),
                       activation='tanh',
                       solver='adam',
                       learning_rate_init=0.0001,
                       max_iter=1000,
                       tol=0.0000001,
                       verbose=True)
    reg.fit(train_x, train_x)
    reconstruction_x = reg.predict(train_x)
    error = np.linalg.norm(train_x - reconstruction_x, axis=1)
    mean = np.mean(error)
    std = np.std(error)

    threshold_dict = {}
    ratio_list = np.array(range(1, 21, 1))
    ratio_list = ratio_list / 10
    ratio_list = list(ratio_list)
    for ratio in ratio_list:
        threshold = mean + ratio * std
        threshold_dict[ratio] = threshold

    with open('threshold.dat', 'wb') as f:
        pickle.dump(threshold_dict, f)

    threshold = mean + 0.4 * std

    return reg, threshold


def AutoEncoderTest(reg, threshold, test_x):
    reconstruction_x = reg.predict(test_x)
    error = np.linalg.norm(test_x - reconstruction_x, axis=1)
    size = len(error)
    predict_y = np.zeros(size)
    for i in range(size):
        if error[i] <= threshold:
            predict_y[i] = 1

    return predict_y


def filter(x, filter_type):
    if filter_type == 'mat3':
        factor = np.ones(10)
        exist = (x > 0) * 1.0
        res = np.dot(exist, factor)
        idx = np.where(res > 2)
        idx = idx[0].tolist()

    if filter_type == 'mat4':
        factor = np.ones(10)
        exist = (x > 0) * 1.0
        res = np.dot(exist, factor)
        idx = np.where(res > 3)
        idx = idx[0].tolist()

    if filter_type == 'mat5':
        factor = np.ones(10)
        exist = (x > 0) * 1.0
        res = np.dot(exist, factor)
        idx = np.where(res > 4)
        idx = idx[0].tolist()

    if filter_type == 'mat6':
        factor = np.ones(10)
        exist = (x > 0) * 1.0
        res = np.dot(exist, factor)
        idx = np.where(res > 5)
        idx = idx[0].tolist()

    if filter_type == 'matY3F3':
        # Y3
        factor = np.ones(5)
        tmp = x[0:, 0:5]
        exist = (tmp > 0) * 1.0
        res = np.dot(exist, factor)
        idx1 = np.where(res > 2)
        # F3
        factor = np.ones(5)
        tmp = x[0:, 5:10]
        exist = (tmp > 0) * 1.0
        res = np.dot(exist, factor)
        idx2 = np.where(res > 2)
        idx = np.append(idx1[0], idx2[0])
        idx = np.unique(idx)

    return x[idx, 0:]


if __name__ == '__main__':
    ppm = 20
    model_type = 'MC'
    filter_type = 'mat3' # 'matY3F3'
    scale_type = 'max'
    file_list = ['正常鼠脑', '去除FUT8的鼠脑']
    # file_list = ['正常鼠脑']
    feature_name = ['N1', 'N2', 'N2H1', 'N2H2', 'N2H3',
                    'N1F1', 'N2F1', 'N2H1F1', 'N2H2F1', 'N2H3F1']

    # Read data from mzML, isotope devolution and charge determination
    for item in file_list:
        filepath_write = item
        folder = os.path.exists(filepath_write)
        if not folder:
            os.makedirs(filepath_write)
        DataPreprocess.get_data_batch(filepath_read=item,
                                      filepath_write=filepath_write)

    # Feature extraction
    num = 0
    for item in file_list:
        filepath_read = item
        filepath_write = item
        folder = os.path.exists(filepath_write)
        if not folder:
            os.makedirs(filepath_write)
        num += 1
        print(str(num) + '/' + str(len(file_list)) + ' ' + item)
        DataPreprocess.extract_feature_batch(filepath_read=filepath_read,
                                      filepath_write=filepath_write, ppm=ppm)

    # Get labeled data, unlabeled data, core_MB data
    path_write = 'train data'
    folder = os.path.exists(path_write)
    if not folder:
        os.makedirs(path_write)
    # labeled data
    path = '去除FUT8的鼠脑'
    dirs = os.listdir(path)
    labeled_x = np.zeros((1, len(feature_name)))
    for file_name in dirs:
        tmp = DataPreprocess.getdata_all(path + '\\' + file_name, feature_name)
        labeled_x = np.concatenate((labeled_x, tmp))
    labeled_x = np.delete(labeled_x, 0, axis=0)
    with open(path_write + '\\labeled_x_' + str(ppm) + 'ppm.dat', 'wb') as f:
        pickle.dump(labeled_x, f)
    # unlabeled data
    path = '正常鼠脑'
    unlabeled_x = np.zeros((1, len(feature_name)))
    file_name = ['MouseBrain_C18_HILIC_8_4h_50cm_IGP_jia_1-1_result.dat',
             'MouseBrain_C18_HILIC_8_4h_50cm_IGP_jia_2_result.dat',
             'MouseBrain_C18_HILIC_8_4h_50cm_IGP_jia_3_result.dat']
    for item in file_name:
        tmp = DataPreprocess.getdata_all(path + '\\' + item, feature_name)
        unlabeled_x = np.concatenate((unlabeled_x, tmp))
    unlabeled_x = np.delete(unlabeled_x, 0, axis=0)
    with open(path_write + '\\unlabeled_x_' + str(ppm) + 'ppm.dat', 'wb') as f:
        pickle.dump(unlabeled_x, f)
    # core_MB data
    path = '正常鼠脑'
    dirs = os.listdir(path)
    core_x_mb = np.zeros((1, len(feature_name)))
    for file_name in dirs:
        tmp = DataPreprocess.getdata_core(path + '\\' + file_name, feature_name, sample_type='MB')
        core_x_mb = np.concatenate((core_x_mb, tmp))
    core_x_mb = np.delete(core_x_mb, 0, axis=0)
    with open(path_write + '\\core_x_mb_' + str(ppm) + 'ppm.dat', 'wb') as f:
        pickle.dump(core_x_mb, f)

    # split train test data
    print('split train test data')
    path_write = 'E:\\python_project\\GlycoPrediction\\result\\train data\\' + averagine
    with open(path_write + '\\labeled_x_' + str(ppm) + 'ppm.dat', 'rb') as f:
        labeled_x = pickle.load(f)
    labeled_x = filter(labeled_x, filter_type=filter_type)
    bound_split = int(labeled_x.shape[0] / 5)
    print('n_test_data: ' + str(bound_split))
    np.random.shuffle(labeled_x)
    labeled_test_x = labeled_x[0:bound_split, 0:]
    labeled_train_x = labeled_x[bound_split:, 0:]
    path_write_test = path_write + '\\' + filter_type
    folder = os.path.exists(path_write_test)
    if not folder:
        os.makedirs(path_write_test)
    with open(path_write_test + '\\labeled_train_x_' + str(ppm) + 'ppm.dat', 'wb') as f:
        pickle.dump(labeled_train_x, f)
    with open(path_write_test + '\\labeled_test_x_' + str(ppm) + 'ppm.dat', 'wb') as f:
        pickle.dump(labeled_test_x, f)

    # train data
    path_write = 'E:\\python_project\\GlycoPrediction\\result\\train data\\' + averagine
    with open(path_write + '\\unlabeled_x_' + str(ppm) + 'ppm.dat', 'rb') as f:
        unlabeled_x = pickle.load(f)
    path_write_test = path_write + '\\' + filter_type
    with open(path_write_test + '\\labeled_train_x_' + str(ppm) + 'ppm.dat', 'rb') as f:
        labeled_x = pickle.load(f)
    labeled_x = filter(labeled_x, filter_type='mat5')
    unlabeled_x = filter(unlabeled_x, filter_type=filter_type)
    print('n_unlabeled: ' + str(unlabeled_x.shape[0]))
    if scale_type == 'sum':
        sum = np.sum(labeled_x, axis=1)
        sum = sum.reshape((sum.shape[0], 1))
        labeled_x = labeled_x/sum
        sum = np.sum(unlabeled_x, axis=1)
        sum = sum.reshape((sum.shape[0], 1))
        unlabeled_x = unlabeled_x/sum
    if scale_type == 'max':
        mas = MaxAbsScaler()
        labeled_x = mas.fit_transform(labeled_x.T)
        labeled_x = labeled_x.T
        unlabeled_x = mas.fit_transform(unlabeled_x.T)
        unlabeled_x = unlabeled_x.T

    # ###
    # # Autoencoder model
    # reg, threshold = AutoEncoder(labeled_x, n_encoder1=9, n_encoder2=8, n_decoder2=8, n_decoder1=9, n_latent=7)
    #
    # ##
    # with open('threshold.dat', 'rb') as f:
    #     threshold_dict = pickle.load(f)
    # ratio_list = np.array(range(1, 21, 1))
    # ratio_list = ratio_list / 10
    # ratio_list = list(ratio_list)
    # ##
    #
    # # unlabel, Fut8 accuracy, LYD core fucosylation accuracy, MB high mannose core fucosylation accuracy result
    # # LYD core fucosylation
    # with open(path_write + '\\core_x_lyd_' + str(ppm) + 'ppm.dat', 'rb') as f:
    #     x_core_fuc = pickle.load(f)
    # factor = np.ones(5)
    # tmp = x_core_fuc[0:, 5:10]
    # exist = (tmp > 0) * 1.0
    # res = np.dot(exist, factor)
    # idx = np.where(res > 2)
    # x_core_fuc = x_core_fuc[idx[0], 0:]
    # if scale_type == 'max':
    #     mas = MaxAbsScaler()
    #     x_core_fuc = mas.fit_transform(x_core_fuc.T)
    #     x_core_fuc = x_core_fuc.T
    # if scale_type == 'sum':
    #     sum = np.sum(x_core_fuc, axis=1)
    #     sum = sum.reshape((sum.shape[0], 1))
    #     x_core_fuc = x_core_fuc / sum
    # result = AutoEncoderTest(reg, threshold, test_x=x_core_fuc)
    # idx = np.where(result == 0)
    # print('LYD core fucosylation num: ' + str(x_core_fuc.shape[0]))
    # print('LYD core fucosylation accuracy: ' + str(len(idx[0]) / len(result)))
    # x_wrong_lyd = x_core_fuc[idx[0].tolist(), 0:]
    # # MB high mannose core fucosylation
    # with open(path_write + '\\core_x_mb_' + str(ppm) + 'ppm.dat', 'rb') as f:
    #     x_core_fuc = pickle.load(f)
    # factor = np.ones(5)
    # tmp = x_core_fuc[0:, 5:10]
    # exist = (tmp > 0) * 1.0
    # res = np.dot(exist, factor)
    # idx = np.where(res > 2)
    # x_core_fuc = x_core_fuc[idx[0], 0:]
    # if scale_type == 'max':
    #     mas = MaxAbsScaler()
    #     x_core_fuc = mas.fit_transform(x_core_fuc.T)
    #     x_core_fuc = x_core_fuc.T
    # if scale_type == 'sum':
    #     sum = np.sum(x_core_fuc, axis=1)
    #     sum = sum.reshape((sum.shape[0], 1))
    #     x_core_fuc = x_core_fuc / sum
    # result = AutoEncoderTest(reg, threshold, test_x=x_core_fuc)
    # idx = np.where(result == 0)
    # print('MB high mannose core fucosylation num: ' + str(x_core_fuc.shape[0]))
    # print('MB high mannose core fucosylation accuracy: ' + str(len(idx[0]) / len(result)))
    # x_wrong_mb = x_core_fuc[idx[0].tolist(), 0:]
    #
    # ##
    # MB_list = []
    # for ratio in threshold_dict:
    #     threshold = threshold_dict[ratio]
    #     result = AutoEncoderTest(reg, threshold, test_x=x_core_fuc)
    #     idx = np.where(result == 0)
    #     accuracy = len(idx[0]) / len(result)
    #     MB_list.append(accuracy)
    # with open('MB_list.dat', 'wb') as f:
    #     pickle.dump(MB_list, f)
    # ##
    #
    # # Fut8 test
    # with open(path_write_test + '\\labeled_test_x_' + str(ppm) + 'ppm.dat', 'rb') as f:
    #     labeled_test_x = pickle.load(f)
    # if scale_type == 'max':
    #     mas = MaxAbsScaler()
    #     labeled_test_x = mas.fit_transform(labeled_test_x.T)
    #     labeled_test_x = labeled_test_x.T
    # if scale_type == 'sum':
    #     sum = np.sum(labeled_test_x, axis=1)
    #     sum = sum.reshape((sum.shape[0], 1))
    #     labeled_test_x = labeled_test_x / sum
    # result = AutoEncoderTest(reg, threshold, test_x=labeled_test_x)
    # idx = np.where(result == 1)
    # print('Fut8 test num: ' + str(len(result)))
    # print('Fut8 test accuracy: ' + str(len(idx[0]) / len(result)))
    # x_wrong_mb_fut8 = labeled_test_x[idx[0].tolist(), 0:]
    # result = AutoEncoderTest(reg, threshold, test_x=unlabeled_x)
    # idx = np.where(result == 0)
    # print('Positive detected: ' + str(len(idx[0])))
    #
    # ##
    # Fut_list = []
    # for ratio in threshold_dict:
    #     threshold = threshold_dict[ratio]
    #     result = AutoEncoderTest(reg, threshold, test_x=labeled_test_x)
    #     idx = np.where(result == 1)
    #     accuracy = len(idx[0]) / len(result)
    #     Fut_list.append(accuracy)
    # with open('Fut_list.dat', 'wb') as f:
    #     pickle.dump(Fut_list, f)
    # l1 = plt.plot(ratio_list, MB_list, 'r--', label='core')
    # l2 = plt.plot(ratio_list, Fut_list, 'g--', label='Non-core')
    # plt.plot(ratio_list, MB_list, 'ro-', ratio_list, Fut_list, 'g+-')
    # # plt.title('The Lasers in Three Conditions')
    # plt.xlabel('k')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()
    # ##
    #
    # print(1)
    # ###

    ###
    # MC model
    filepath_model = path_write + '\\' + model_type + '_' + filter_type + \
                     '_' + str(ppm) + 'ppm_' + scale_type + '.model'
    MappingConvergence(labeled_x, unlabeled_x, filepath_model=filepath_model)

    # test
    clf = joblib.load(filepath_model)
    # LYD core fucosylation accuracy
    with open(path_write + '\\core_x_lyd_' + str(ppm) + 'ppm.dat', 'rb') as f:
        x_core_fuc = pickle.load(f)
    factor = np.ones(5)
    tmp = x_core_fuc[0:, 5:10]
    exist = (tmp > 0) * 1.0
    res = np.dot(exist, factor)
    idx = np.where(res > 2)
    x_core_fuc = x_core_fuc[idx[0], 0:]
    if scale_type == 'max':
        mas = MaxAbsScaler()
        x_core_fuc = mas.fit_transform(x_core_fuc.T)
        x_core_fuc = x_core_fuc.T
    if scale_type == 'sum':
        sum = np.sum(x_core_fuc, axis=1)
        sum = sum.reshape((sum.shape[0], 1))
        x_core_fuc = x_core_fuc / sum
    result = clf.predict(x_core_fuc)
    idx = np.where(result==0)
    print('LYD core fucosylation num: ' + str(x_core_fuc.shape[0]))
    print('LYD core fucosylation accuracy: ' + str(len(idx[0]) / len(result)))
    x_wrong_lyd = x_core_fuc[idx[0].tolist(), 0:]
    # MB high mannose core fucosylation accuracy
    with open(path_write + '\\core_x_mb_' + str(ppm) + 'ppm.dat', 'rb') as f:
        x_core_fuc = pickle.load(f)
    factor = np.ones(5)
    tmp = x_core_fuc[0:, 5:10]
    exist = (tmp > 0) * 1.0
    res = np.dot(exist, factor)
    idx = np.where(res > 2)
    x_core_fuc = x_core_fuc[idx[0], 0:]
    if scale_type == 'max':
        mas = MaxAbsScaler()
        x_core_fuc = mas.fit_transform(x_core_fuc.T)
        x_core_fuc = x_core_fuc.T
    if scale_type == 'sum':
        sum = np.sum(x_core_fuc, axis=1)
        sum = sum.reshape((sum.shape[0], 1))
        x_core_fuc = x_core_fuc / sum
    result = clf.predict(x_core_fuc)
    idx = np.where(result == 0)
    print('MB high mannose core fucosylation num: ' + str(x_core_fuc.shape[0]))
    print('MB high mannose core fucosylation accuracy: ' + str(len(idx[0]) / len(result)))
    x_wrong_mb = x_core_fuc[idx[0].tolist(), 0:]
    # Fut8 test
    with open(path_write_test + '\\labeled_test_x_' + str(ppm) + 'ppm.dat', 'rb') as f:
        labeled_test_x = pickle.load(f)
    if scale_type == 'max':
        mas = MaxAbsScaler()
        labeled_test_x = mas.fit_transform(labeled_test_x.T)
        labeled_test_x = labeled_test_x.T
    if scale_type == 'sum':
        sum = np.sum(labeled_test_x, axis=1)
        sum = sum.reshape((sum.shape[0], 1))
        labeled_test_x = labeled_test_x / sum
    result = clf.predict(labeled_test_x)
    idx = np.where(result == 1)
    print('Fut8 test num: ' + str(len(result)))
    print('Fut8 test accuracy: ' + str(len(idx[0]) / len(result)))
    x_wrong_mb_fut8 = labeled_test_x[idx[0].tolist(), 0:]
    result = clf.predict(unlabeled_x)
    idx = np.where(result == 0)
    print('Positive detected: ' + str(len(idx[0])))
    # f1 score
    train_x = np.concatenate((labeled_x, unlabeled_x[idx[0].tolist(), 0:]))
    train_y = np.concatenate((np.ones((labeled_x.shape[0], 1)), np.zeros((len(idx[0]), 1))))
    result = clf.predict(train_x)
    tp = 0
    for file_name in range(len(result)):
        if result[file_name] and train_y[file_name]:
            tp += 1
    recall = tp / np.sum(train_y)
    precision = tp / np.sum(result)
    print('F1 score: ' + str(2 * (recall * precision) / (recall + precision)))
    print(1)
    ###
