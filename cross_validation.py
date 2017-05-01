import warnings
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.cross_validation import StratifiedKFold
from LinearAlg import NaiveBayes, SindromAlgorithm
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')
def cross_val_score(
              model,
              data_dir,
              klist=[44],
              bin_thrlist=[2],
              n_shuffles=10,
              n_folds=10,
              type_weights=1):

    scores = {}
    scores['AUC(Train)'] = {}
    scores['AUC(Test)'] = {}
    scores['LogLoss(Train)'] = {}
    scores['LogLoss(Test)'] = {}
    for n_shuffle in range(n_shuffles):
        X_train = np.load('{}/vw_train{}.npy'.format(data_dir, n_shuffle))
        y_train = np.load('{}/true_labels_train{}.npy'.format(data_dir, n_shuffle))
        n_observs = np.load('{}/observs_train{}.npy'.format(data_dir, n_shuffle))
                
        _, idx = np.unique(n_observs, return_index=True)
        y_train_observs = y_train[np.sort(idx)]
        skf = StratifiedKFold(y_train_observs, n_folds, random_state=241)
        
        for k in klist:
            for bin_thr in bin_thrlist:
                key = (k, bin_thr)
                scores['AUC(Train)'][key] = []
                scores['AUC(Test)'][key] = []
                scores['LogLoss(Train)'][key] = []
                scores['LogLoss(Test)'][key] = []
                for n_fold, (i, j) in enumerate(skf):
                    train_idx = np.in1d(n_observs, n_observs[i])
                    test_idx = np.in1d(n_observs, n_observs[j])
                    if model.__name__ == 'SindromAlgorithm':
                        clf = CalibratedClassifierCV(SindromAlgorithm(k, bin_thr, type_weights)).fit(
                                                                        X_train[train_idx], y_train[train_idx])
                    else:
                        clf = CalibratedClassifierCV(NaiveBayes(k)).fit(X_train[train_idx], y_train[train_idx])
                        
                    probas_train = clf.predict_proba(X_train[train_idx])[:, 1]
                    probas_test = clf.predict_proba(X_train[test_idx])[:, 1]

                    scores['AUC(Train)'][key].append(roc_auc_score(y_train[train_idx], probas_train))
                    scores['AUC(Test)'][key].append(roc_auc_score(y_train[test_idx], probas_test))
                    scores['LogLoss(Train)'][key].append(log_loss(y_train[train_idx], probas_train))
                    scores['LogLoss(Test)'][key].append(log_loss(y_train[test_idx], probas_test))
           
    mean_scores = {}
    mean_scores['AUC(Train)'] = {}
    mean_scores['AUC(Test)'] = {}
    mean_scores['LogLoss(Train)'] = {}
    mean_scores['LogLoss(Test)'] = {}

    for key in scores['AUC(Train)'].keys():
        mean_scores['AUC(Train)'][key] = np.mean(scores['AUC(Train)'][key])
        mean_scores['AUC(Test)'][key] = np.mean(scores['AUC(Test)'][key])
        mean_scores['LogLoss(Train)'][key] = np.mean(scores['LogLoss(Train)'][key])
        mean_scores['LogLoss(Test)'][key] = np.mean(scores['LogLoss(Test)'][key])
        
    best_key = max(mean_scores['AUC(Test)'], key=lambda i: mean_scores['AUC(Test)'][i])
    return (best_key,
            mean_scores)


