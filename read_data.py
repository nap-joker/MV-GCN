# import pickle
#
# # 打开.pkl文件，使用'rb'模式表示以二进制只读方式打开
# with open('./data/dti.coo.pkl', 'rb') as f:
#     # 使用pickle.load方法加载数据
#     data = pickle.load(f)
#
# # 此时data变量就存储了从.pkl文件中读取出来的数据
# # 可以根据数据的具体类型（比如字典、列表等）进行后续相应的操作
# print(data[:,0,0]) #shape为599*84*3
""" Code for loading data. """
import sklearn, sklearn.datasets
import sklearn.naive_bayes, sklearn.linear_model, sklearn.svm, sklearn.neighbors, sklearn.ensemble
import matplotlib.pyplot as plt
import scipy.sparse
import numpy as np
import time, re
import pickle as pkl
import hickle as hkl
import pandas as pd
import scipy.io as sio

from sklearn.model_selection import StratifiedKFold

# bad_mri_id = [523, 524, 639, 643, 647, 767]

def load_mri():#def load_dti(data_type): # not that make sense, but still can be try
    # delete_sid = [373] + bad_mri_id # dti hough do not have the id 373
    subj = list()#存储被试者编号
    data = list()#存储具体的数据内容
    # filepath = '../../data/ppmi/input/dti.roi/' + data_type[0]
    # sid=pd.read_csv('E:/Graduation_design_in_2024/liu/ABCD-data/mh_ksads_sums_diag_innerjoined_diag_lable_2kinds_100Q.csv').to_numpy()[:,0]
    # sid=np.ravel(sio.loadmat('../ABCD-data/idx.mat').get('idx'))#第1282列，python从0开始索引
    sid = np.array([i for i in range(3549)])
    for i in range(3549):
        data_v = list()
        # if i in delete_sid:
        #     continue

        for view in range(1):
            # filepath = '../../data/ppmi/input/dti.roi/' + view
            if (i+1) not in subj:
                subj.append((i+1))
            # print("reading connectivity file %s" % i)
            if view == 0:
                try:
                    mat=sio.loadmat('../msn_fc_data.mat')['msn_matrix'][:,:,i]
                # mat = sio.loadmat(filepath + '_' + str(i) + '.mat')['A']
                    data_v.append(np.array(mat))
                # if np.sum(np.array(mat, dtype='int32')) == 0:
                #     print (i)
                #     print (view)
                except IOError:
                    data_v.append(np.zeros([68, 68]))
                    print("File  does not exit")
            if view == 1:
                try:
                    mat=sio.loadmat('../msn_fc_data.mat')['fc_matrix'][:,:,i]
                # mat = sio.loadmat(filepath + '_' + str(i) + '.mat')['A']
                    data_v.append(np.array(mat))
                # if np.sum(np.array(mat, dtype='int32')) == 0:
                #     print (i)
                #     print (view)
                except IOError:
                    data_v.append(np.zeros([68, 68]))
                    print("File  does not exit")
        data.append(data_v)
    return data, subj

def load_roi_coords():
    coords = pd.read_excel('DK68cood.xlsx').to_numpy()
    coords = coords[:,:3]
    coords = np.repeat(coords[np.newaxis, :, :], 100, axis=0)
    return coords


def load_data(valid_portion=0.1, test_portion=0.1, kfold='False'):
    """Load data."""
    # print (data_type)
    # load pairs
    # f = open('../../data/ppmi/input/dti.pd.pairs.pkl', 'rb')
    pairs = pd.DataFrame(sio.loadmat('../msn_fc_data.mat')['pairs']).iloc[:,0:2]
    pairs = pairs.values.tolist()
    pairs = [[element - 1 for element in sublist] for sublist in pairs]
    labels = pd.DataFrame(sio.loadmat('../msn_fc_data.mat')['pairs']).iloc[:,2]
    # f.close()
    sid = [i[0] for i in pairs]
    a = np.unique(np.array(sid))
    sio.savemat('subject_id.mat', {'subj': a})

    # load roi coordinates
    coords = load_roi_coords()
    # print(coords.shape)

    # load data
    data, subj = load_mri() # dictionary for multiview
    data = np.array(data)

    # train, validate, test split
    if kfold == 'False':
        n_pairs = len(pairs)
        n_samples = len(subj)
        sidx = np.random.permutation(n_pairs)
        n_train = int(np.round(n_pairs * (1. - valid_portion - test_portion)))
        # val_pairs = [pairs[s] for s in sidx[:n_train]] #这是一个bug叭？
        val_pairs = [pairs[s] for s in sidx[n_train:]]
        val_labels = [labels[s] for s in sidx[n_train:]]
        train_pairs = [pairs[s] for s in sidx[:n_train]]
        train_labels = [labels[s] for s in sidx[:n_train]]

        sidx = np.random.permutation((n_pairs-n_train))
        n_val = int(np.round((n_samples-n_train) * (valid_portion/(valid_portion+test_portion))))
        test_pairs = [val_pairs[s] for s in sidx[n_val:]]
        test_labels = [val_labels[s] for s in sidx[n_val:]]
        val_pairs = [val_pairs[s] for s in sidx[:n_val]]
        val_labels = [val_labels[s] for s in sidx[:n_val]]

        train_pairs = np.array(train_pairs)
        val_pairs = np.array(val_pairs)
        test_pairs = np.array(test_pairs)
        train_labels = np.array(train_labels)
        val_labels = np.array(val_labels)
        test_labels = np.array(test_labels)
        pairs_set = (train_pairs, val_pairs, test_pairs)
        labels_set = (train_labels, val_labels, test_labels)

    elif kfold=='True':
        pairs = np.array(pairs)
        labels = np.array(labels)
        skf = StratifiedKFold(n_splits=5)
        pairs_set = list()
        labels_set = list()
        for train_index, test_index in skf.split(pairs, labels):
            train_x, test_x = pairs[train_index], pairs[test_index]
            train_y, test_y = labels[train_index], labels[test_index]
            val_x = test_x
            val_y = test_y
            pairs_set.append((train_x, val_x, test_x))
            labels_set.append((train_y, val_y, test_y))
    return data, subj, coords, pairs_set, labels_set


### Helpers to quantify classifier's quality.
def baseline(train_data, train_labels, test_data, test_labels, omit=[]):
    """Train various classifiers to get a baseline."""
    clf, train_accuracy, test_accuracy, train_f1, test_f1, exec_time = [], [], [], [], [], []
    clf.append(sklearn.neighbors.KNeighborsClassifier(n_neighbors=10))
    clf.append(sklearn.linear_model.LogisticRegression())
    clf.append(sklearn.naive_bayes.BernoulliNB(alpha=.01))
    clf.append(sklearn.ensemble.RandomForestClassifier())
    clf.append(sklearn.naive_bayes.MultinomialNB(alpha=.01))
    clf.append(sklearn.linear_model.RidgeClassifier())
    clf.append(sklearn.svm.LinearSVC())
    for i,c in enumerate(clf):
        if i not in omit:
            t_start = time.process_time()
            c.fit(train_data, train_labels)
            train_pred = c.predict(train_data)
            test_pred = c.predict(test_data)
            train_accuracy.append('{:5.2f}'.format(100*sklearn.metrics.accuracy_score(train_labels, train_pred)))
            test_accuracy.append('{:5.2f}'.format(100*sklearn.metrics.accuracy_score(test_labels, test_pred)))
            train_f1.append('{:5.2f}'.format(100*sklearn.metrics.f1_score(train_labels, train_pred, average='weighted')))
            test_f1.append('{:5.2f}'.format(100*sklearn.metrics.f1_score(test_labels, test_pred, average='weighted')))
            exec_time.append('{:5.2f}'.format(time.process_time() - t_start))
    print('Train accuracy:      {}'.format(' '.join(train_accuracy)))
    print('Test accuracy:       {}'.format(' '.join(test_accuracy)))
    print('Train F1 (weighted): {}'.format(' '.join(train_f1)))
    print('Test F1 (weighted):  {}'.format(' '.join(test_f1)))
    print('Execution time:      {}'.format(' '.join(exec_time)))

def grid_search(params, grid_params, train_data, train_labels, val_data,
        val_labels, test_data, test_labels, model):
    """Explore the hyper-parameter space with an exhaustive grid search."""
    params = params.copy()
    train_accuracy, test_accuracy, train_f1, test_f1 = [], [], [], []
    grid = sklearn.grid_search.ParameterGrid(grid_params)
    print('grid search: {} combinations to evaluate'.format(len(grid)))
    for grid_params in grid:
        params.update(grid_params)
        name = '{}'.format(grid)
        print('\n\n  {}  \n\n'.format(grid_params))
        m = model(params)
        m.fit(train_data, train_labels, val_data, val_labels)
        string, accuracy, f1, loss = m.evaluate(train_data, train_labels)
        train_accuracy.append('{:5.2f}'.format(accuracy)); train_f1.append('{:5.2f}'.format(f1))
        print('train {}'.format(string))
        string, accuracy, f1, loss = m.evaluate(test_data, test_labels)
        test_accuracy.append('{:5.2f}'.format(accuracy)); test_f1.append('{:5.2f}'.format(f1))
        print('test  {}'.format(string))
    print('\n\n')
    print('Train accuracy:      {}'.format(' '.join(train_accuracy)))
    print('Test accuracy:       {}'.format(' '.join(test_accuracy)))
    print('Train F1 (weighted): {}'.format(' '.join(train_f1)))
    print('Test F1 (weighted):  {}'.format(' '.join(test_f1)))
    for i,grid_params in enumerate(grid):
        print('{} --> {} {} {} {}'.format(grid_params, train_accuracy[i], test_accuracy[i], train_f1[i], test_f1[i]))
        

#data, subj, coords, pairs_set, labels_set=load_data(valid_portion=0.1, test_portion=0.1, kfold='False')
