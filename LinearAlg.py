import numpy as np
from sklearn.base import BaseEstimator

class NaiveBayes(BaseEstimator):
    def __init__(self, k=216):
        self.k = k

    def fit(self, X, y):
        p_cw = np.zeros((2, self.k))
        p_cw[0, :] = X[y == 0].sum(axis=0) 
        p_cw[1, :] = X[y == 1].sum(axis=0)  
        p_cw /= X.sum(axis=0) 
        p_cw[np.isnan(p_cw)] = 0.5
        self.p_cw = p_cw.copy()
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        probas = self.p_cw.dot(X.T).T
        return probas

class SindromAlgorithm(BaseEstimator):
    def __init__(self, k=44, bin_thr=2, type_weights=1):
        self.k = k
        self.bin_thr = bin_thr
        self.type_weights = type_weights 

    def fit(self, X, y):
        X_bin = (X >= self.bin_thr).astype(int)
        n_dis = np.count_nonzero(y)
        sort_values_dis = (X_bin[y == 1].sum(
            axis=0) + 1.0) / (n_dis + 2.0)
        sort_values_health = (X_bin[y == 0].sum(
            axis=0) + 1.0) / (y.shape[0] - n_dis + 2.0)

        if self.type_weights == 1:
            weights = np.log(sort_values_dis * (1 - sort_values_health)) - \
                np.log(sort_values_health * (1 - sort_values_dis))
        else:
            weights = np.mean(X_bin[y == 1], axis=0) - np.mean(X_bin[y == 0], axis=0)
        selected_feat = np.argsort(sort_values_dis)[::-1][:self.k]
            
        final_weights = np.zeros(weights.shape)
        final_weights[selected_feat] = weights[selected_feat]
        self.final_weights = final_weights.copy()
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        X_bin = (X >= self.bin_thr).astype(int)
        probas = X_bin.dot(self.final_weights.reshape(self.final_weights.shape[0], 1))
        return np.hstack((1 - probas, probas))