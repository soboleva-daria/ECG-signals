import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

def naive_bayes(X_train, y_train, X_test, y_test, k=216):

    p_cw = np.zeros((2, k))
    p_cw[0, :] = X_train[y_train == 0].sum(axis=0) 
    p_cw[1, :] = X_train[y_train == 1].sum(axis=0)  
    p_cw /= X_train.sum(axis=0) 
    p_cw[np.isnan(p_cw)] = 0.5
    
    probas_train = p_cw.dot(X_train.T) 
    probas_test = p_cw.dot(X_test.T) 
    
    predict_train = probas_train[1] 
    predict_train /= predict_train.max() or 1
    
    predict_test = probas_test[1] 
    predict_test /= predict_test.max() or 1

    return (roc_auc_score(y_train, predict_train),
            roc_auc_score(y_test, predict_test),
            log_loss(y_train, predict_train),
            log_loss(y_test, predict_test))


def naive_bayes_bin(X_train, y_train, X_test, y_test, k=216, bin_thr=2):
    X_train_bin = (X_train >= bin_thr).astype(int)
    X_test_bin = (X_test >= bin_thr).astype(int)
    return naive_bayes(
               X_train_bin,
               y_train, 
               X_test_bin, 
               y_test)


def sindrom_algorithm(X_train, y_train, X_test, y_test, k = 172, bin_thr=2, type_weights=1):

    X_train_bin = (X_train >= bin_thr).astype(int)
    X_test_bin = (X_test >= bin_thr).astype(int)
    
    n_dis = np.count_nonzero(y_train)

    sort_values_dis = (X_train_bin[y_train == 1].sum(
        axis=0) + 1.0) / (n_dis + 2.0)
    sort_values_health = (X_train_bin[y_train == 0].sum(
        axis=0) + 1.0) / (y_train.shape[0] - n_dis + 2.0)

    selected_feat = np.argsort(sort_values_dis)[::-1][:k]

    if type_weights == 1:
        weights = np.log(sort_values_dis * (1 - sort_values_health)) - \
            np.log(sort_values_health * (1 - sort_values_dis))
    else:
        weights = np.mean(X_train_bin[y_train == 1], axis=0) - np.mean(X_train_bin[y_train == 0], axis=0)
        
    final_weights = np.zeros(weights.shape)
    final_weights[selected_feat] = weights[selected_feat]

    predict_train = X_train_bin.dot(final_weights.reshape(final_weights.shape[0], 1))
    predict_train /= predict_train.max() or 1
    
    predict_test = X_test_bin.dot(final_weights.reshape(final_weights.shape[0], 1))
    predict_test /= predict_test.max() or 1
    
    return (roc_auc_score(y_train, predict_train),
            roc_auc_score(y_test, predict_test),
            log_loss(y_train, predict_train),
            log_loss(y_test, predict_test))