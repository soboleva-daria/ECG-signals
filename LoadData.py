import numpy as np
import pandas as pd
from collections import OrderedDict
import os 

def create_train_vw(data_all, target, file_res, file_true_labels, file_observs_train, predictors):
    n_observs = data_all.n_observ
    X_train = data_all[np.setdiff1d(data_all.columns, [target])][predictors]
    y_train = data_all[target]
    
    dis_idxs = y_train[y_train == 1].index
    data = X_train.T.apply(OrderedDict)

    with open(file_true_labels + '.txt', 'w') as lab:
        with open(file_res + '.txt', 'w') as fout:
            for key, value  in zip(data.index, data):
                fout.write('{} '.format(n_observs.loc[key]))

                for tr, freq in value.items():
                    fout.write('{}:{} '.format(tr, freq))

                fout.write('|labels ')

                if key in dis_idxs:
                    fout.write('{}:{} '.format('label0', 0))
                    fout.write('{}:{}'.format(' label1', 1))
                    lab.write('1')
                else:
                    fout.write('{}:{}'.format('label0', 1))
                    fout.write('{}:{}'.format(' label1', 0))
                    lab.write('0')

                fout.write('\n')
                lab.write('\n')
    np.save(file_res + '.npy', X_train.values)
    np.save(file_true_labels + '.npy', y_train.values) 
    np.save(file_observs_train + '.npy', n_observs)
    
def create_valid_vw(data_all, target, file_res, file_true_labels, predictors):
    n_observs = data_all.n_observ
    X_valid = data_all[np.setdiff1d(data_all.columns, [target])][predictors]
    y_valid = data_all[target]
    
    dis_idxs = y_valid[y_valid == 1].index
    data = X_valid[predictors].T.apply(OrderedDict)

    with open(file_true_labels + '.txt', 'w') as lab:
        with open(file_res + '.txt', 'w') as fout:
            for key, value  in zip(data.index, data):
                fout.write('{} '.format(n_observs.loc[key]))

                for tr, freq in value.items():
                    fout.write('{}:{} '.format(tr, freq))

                #fout.write('|labels ')

                if key in dis_idxs:
                    lab.write('1')
                else:
                    lab.write('0')

                fout.write('\n')
                lab.write('\n')
    np.save(file_res + '.npy', X_valid.values)
    np.save(file_true_labels + '.npy', y_valid.values)


def reject_shuffle(idxs, n_shuffles):
    orders = set()       
    for i in range(n_shuffles):
        while tuple(idxs) in orders:
            np.random.shuffle(idxs)
        orders.add(tuple(idxs))
    for idx in orders:
        yield idx

def create_shuffled_vws(file_src, file_true_labels, file_observs_train, n_shuffles):
    with open(file_src + '.txt', 'r') as fin:
        data = fin.readlines()
    data_arr = np.load(file_src + '.npy')

    with open(file_true_labels + '.txt', 'r') as fin:
        y_true = fin.readlines()
    y_true_arr = np.load(file_true_labels + '.npy')
    n_observs = np.load(file_observs_train + '.npy')
    
    idxs = np.arange(data_arr.shape[0])
    for i, idx in enumerate(reject_shuffle(idxs, n_shuffles)):
        idx_val = np.array(idx, dtype=int)
        data_shuffle = np.array(data)[idx_val]
        y_true_shuffle = np.array(y_true)[idx_val]
        
        data_arr_shuffle = data_arr[idx_val]
        y_true_arr_shuffle = y_true_arr[idx_val]
        n_observs_shuffle = n_observs[idx_val]
    
        with open('{}{}.txt'.format(file_src, i), 'w') as fout:
            fout.write(''.join(data_shuffle))
        np.save('{}{}.npy'.format(file_src, i), data_arr_shuffle)
        
        with open('{}{}.txt'.format(file_true_labels, i), 'w') as fout:
            fout.write(''.join(y_true_shuffle))
        np.save('{}{}.npy'.format(file_true_labels, i), y_true_arr_shuffle)
        np.save('{}{}.npy'.format(file_observs_train, i), n_observs_shuffle)


def read_data(n_dis, n_health, predictors): 
    data = pd.read_csv('data.txt', sep='\t')
    data.columns = ['n_record', 'n_observ'] + list(range(1, 137))

    p3 = pd.read_csv('p3.txt', sep='\t')
    p3.columns  = predictors

    ind_health = data[n_health] == 1
    ind_dis = data[n_dis] == 1

    data_health = p3.loc[ind_health]
    data_dis = p3.loc[ind_dis]

    data_all = pd.concat([data_health, data_dis], axis=0)
    data_all['n_observ'] = data.n_observ

    target = 'p_diseas'
    data_all[target] = 0
    data_all.loc[ind_dis, target]  = 1
    return data_all, target

def train_test_split(data, predictors, target, data_dir, train_frac=0.7, n_shuffles=10):
    data_all = data.sample(frac=1, random_state=241)
    observs = data.n_observ.unique()
    train_size = int(observs.shape[0] * train_frac) + 1
    np.random.seed(241)
    train_observs = np.random.choice(
                       observs,
                       train_size,
                       replace=False)

    train_idxs = data_all.n_observ.isin(train_observs)
    train = data_all[train_idxs]
    valid = data_all[~train_idxs]

    X_train = train[predictors]
    y_train = train[target]

    X_valid = valid[predictors]
    y_valid = valid[target]

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    create_train_vw(
        train,
        target,
        '{}/vw_train'.format(data_dir),
        '{}/true_labels_train'.format(data_dir),
        '{}/observs_train'.format(data_dir),
        predictors
    )
    create_valid_vw(
        valid,
        target,
        '{}/vw_valid'.format(data_dir),
        '{}/true_labels_valid'.format(data_dir),
        predictors)

    create_shuffled_vws(
        '{}/vw_train'.format(data_dir),
        '{}/true_labels_train'.format(data_dir),
        '{}/observs_train'.format(data_dir),
        n_shuffles)

    return X_train, y_train, X_valid, y_valid