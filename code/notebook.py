import pandas as pd
import numpy as np
import time

from LoadData import read_data, train_test_split
from ARTM import EcgClassification
from LinearAlg import SindromAlgorithm, NaiveBayes
from cross_validation import cross_val_score
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from statsmodels.stats.weightstats import DescrStatsW
import matplotlib.pyplot as plt

def plot_fill_between(data, data_dir, label, n_dis, color='magenta'):
    plt.figure(figsize=(10, 8))
    left_edge = []
    right_edge = []
    mean_res = []
    iters = []
    for key, value in data.items():
        left, right = DescrStatsW(value).tconfint_mean()
        left_edge.append(left)
        right_edge.append(right)
        iters.append(key + 1)
        mean_res.append(np.mean(value))

    plt.fill_between(iters, left_edge, right_edge, color='violet')
    plt.plot(iters, mean_res, color=color, lw=5)

    plt.xlabel('iteration', fontsize=18)
    plt.ylabel(label, fontsize=18)
    plt.xlim([1, len(iters)])
    plt.ylim([min(mean_res) - 0.05, max(mean_res) + 0.05])
    plt.xticks(list(plt.xticks()[0][1:]) + [1])
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(['TestCV'], fontsize=18, loc=2)
    plt.savefig('{}/{}:{}_ARTM_smart.eps'.format(data_dir, label, n_dis))
    plt.show()
    return mean_res

def plot_artm_res(data_dir, n_dis, auc, logloss, sparsity_phi_c, sparsity_phi_gram3, color='magenta'):
    mean_auc = plot_fill_between(auc, data_dir, 'AUC', n_dis)
    mean_logloss = plot_fill_between(logloss, data_dir, 'LogLoss', n_dis)

    plt.figure(figsize=(10, 8))
    iters = []
    mean_sparsity_phi_c = []
    mean_sparsity_phi_gram3 = []

    for key, value in sparsity_phi_c.items():
        iters.append(key + 1)
        mean_sparsity_phi_c.append(np.mean(value))

    for key, value in sparsity_phi_gram3.items():
        mean_sparsity_phi_gram3.append(np.mean(value))

    plt.plot(iters, mean_sparsity_phi_c, color='magenta', lw=5)
    plt.plot(iters, mean_sparsity_phi_gram3, color='purple', lw=5)

    plt.xlabel('iteration', fontsize=18)
    plt.xlim([1, len(iters)])
    #plt.ylim([0.0 - 0.05, 0.1 + 0.05])

    plt.xticks(list(plt.xticks()[0][1:]) + [1])
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.ylabel('Sparsity', fontsize=18)
    plt.legend([r'$p(c|t)$', r'$p(gram3|t)$'], loc = 'best', fontsize = 15)
    plt.savefig('{}/Sparsity:{}_ARTM_smart.eps'.format(data_dir, n_dis))
    plt.show()
    return mean_auc[-1], mean_logloss[-1], mean_sparsity_phi_c[-1], mean_sparsity_phi_gram3[-1]


n_health = 1
predictors = np.load('predictors.npy')
auc_res = []
logloss_res = [] 
sa_auc = []
sa_logloss = []
nb_auc = []
nb_logloss = []
scores = {}
for n_dis in np.load('disease_names.npy'):
    print (n_dis)
    data_dir = str(n_dis)
    scores[n_dis] = {}
    data_all, target = read_data(n_dis, n_health, predictors)
    X_train, y_train, X_valid, y_valid = train_test_split(data_all, predictors, target, data_dir)

    #clf = EcgClassification(data_all.shape[0], data_dir)
    #search_params = {
    #            'n_topics':range(2, 10, 2), 
    #            'init_methods':['KMeans', 'Random', 'NaiveKMeans', 'Spread', 'KMeansGram3', 'HarmonyBaskets'],
    #            'tau' : [1e2, 1e3, 1e3 * 5],
    #            'tau_phi_gram3': [-1e1, -1e2],
    #            'tau_phi_gram3_decorr':[0.0, 1e0, 1e1, 1e2],
    #        }
    #start_time = time.time()
    
    #clf.find_best_params(search_params)
    #artm_scores, ptc, _ = clf.cross_val_score(shuffles=np.arange(10))
    #scores[n_dis]['time'] = time.time() - start_time
    #res = plot_artm_res(data_dir, n_dis, artm_scores['AUC(Test)'], artm_scores['LogLoss(Test)'], artm_scores['SparsityPhiC'], artm_scores['SparsityPhiGram3'])
    
    #print ('AUC(Test):{}'.format(res[0]),
    #      'LogLoss(Test):{}'.format(res[1]))
    #pd.DataFrame(ptc).to_csv('{}/Ptc:{}_ARTM.csv'.format(data_dir, n_dis), index_col=0)
    #artm_valid_scores = clf.valid_score('true_labels_train.txt', 'true_labels_valid.txt')
    #print(artm_valid_scores['AUC(Test)'][-1], artm_valid_scores['LogLoss(Test)'][-1])
    #auc_res.append(artm_valid_scores['AUC(Test)'][-1])
    #logloss_res.append(artm_valid_scores['LogLoss(Test)'][-1])
    #print(artm_valid_scores['SparsityPhiC'][-1])
    #print(artm_valid_scores['SparsityPhiGram3'][-1])
    #res = cross_val_score(SindromAlgorithm, data_dir, type_weights=1)
    #print(res[1])
    #print(res[2])
    y_pred = CalibratedClassifierCV(SindromAlgorithm()).fit(X_train, y_train).predict_proba(X_valid)[:, 1]
    sa_auc.append(roc_auc_score(y_valid, y_pred))
    sa_logloss.append(log_loss(y_valid, y_pred))
    print(sa_auc[-1], sa_logloss[-1])
    print('---------')

    #res = cross_val_score(NaiveBayes, data_dir, klist=[216])
    #print(res[1])
    #print(res[2])
    y_pred = CalibratedClassifierCV(NaiveBayes()).fit(X_train, y_train).predict_proba(X_valid)[:, 1]
    nb_auc.append(roc_auc_score(y_valid, y_pred))
    nb_logloss.append(log_loss(y_valid, y_pred))
    print(nb_auc[-1])
    print(nb_logloss[-1])
    print('----------')
#pd.DataFrame(scores).to_csv('scores.csv')
np.save('SA:AUC.npy', sa_auc)
np.save('SA:LogLoss.npy', sa_logloss)

np.save('NB:AUC.npy', nb_auc)
np.save('NB:LogLoss.npy', nb_logloss)