import shutil
import string
from itertools import product

from sklearn.metrics import roc_auc_score, log_loss
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pandas as pd

from InitializePhi import initialize_phi
# Configure logging folder
import sys
import os
sys.path.append('../bigartm/python')
import artm
os.environ["ARTM_SHARED_LIBRARY"] = "../bigartm/build/lib/libartm.so"

lc = artm.messages.ConfigureLoggingArgs()
lc.log_dir = 'tmp'
lib = artm.wrapper.LibArtm(logging_config=lc)

# Change any other logging parameters at runtime (except logging folder)
#lc.minloglevel=0  # 0 = INFO, 1 = WARNING, 2 = ERROR, 3 = FATAL
#lib.ArtmConfigureLogging(lc)

EPS = 1e-4
class EcgClassification(object):

    def __init__(self, n_objs, data_dir, process_dir='ecg'):
    
        self.n_objs = n_objs
        self.data_dir = data_dir
        self.process_dir = process_dir
        self.c = 'labels'
        self.gram3 = '@default_class'
        self.trigrams = np.load('predictors.npy')

    def build_cv_folds(self, n_shuffle):
        data_dir = self.data_dir
        with open('{}/true_labels_train{}.txt'.format(data_dir, n_shuffle), 'r') as fin:
            y_true = np.array([int(i) for i in fin.readlines()], dtype=str)
        y_true_arr = np.load('{}/true_labels_train{}.npy'.format(data_dir, n_shuffle))

        with open('{}/vw_train{}.txt'.format(data_dir, n_shuffle), 'r') as fin:
            data = np.array(fin.readlines())
        data_arr = np.load('{}/vw_train{}.npy'.format(data_dir, n_shuffle))

        n_observs = np.load('{}/observs_train{}.npy'.format(data_dir, n_shuffle))

        _, idx = np.unique(n_observs, return_index=True)
        y_true_observs = y_true[np.sort(idx)]
        skf = StratifiedKFold(y_true_observs, self.n_folds, random_state=241)

        process_dir = self.process_dir
        for n_fold, (i, j) in enumerate(skf):
            train_idx = np.in1d(n_observs, n_observs[i])
            test_idx = np.in1d(n_observs, n_observs[j])

            d_dir = '{}/data{}'.format(process_dir, n_fold)
            os.mkdir(d_dir)
            with open(os.path.join(d_dir, 'test.txt'), 'w') as test:
                with open(os.path.join(d_dir, 'train.txt'), 'w') as train:
                    with open(os.path.join(d_dir, 'true_labels_test.txt'), 'w') as test_labels:
                        with open(os.path.join(d_dir, 'true_labels_train.txt'), 'w') as train_labels:

                            train.write(''.join(data[train_idx]))
                            np.save(os.path.join(d_dir, 'train.npy'), data_arr[train_idx])

                            train_labels.write('\n'.join(y_true[train_idx]))
                            np.save(os.path.join(d_dir, 'true_labels_train.npy'), y_true_arr[train_idx])

                            for line in data[test_idx]:
                                test.write("%s\n" % (line.partition('|')[0]))

                            test_labels.write('\n'.join(y_true[test_idx]))


    def create_batches(self, d_dir):
        new_folder = os.path.join(d_dir, 'data_batches')
        target_folder_train = "%s%s" % (new_folder, '_train')
        target_folder_test = "%s%s" % (new_folder, '_test')

        batch_vectorizer = artm.BatchVectorizer(
            batch_size=100000,
            data_path=os.path.join(
                d_dir,
                'train.txt'),
            data_format='vowpal_wabbit',
            target_folder=target_folder_train)

        batch_vectorizer = artm.BatchVectorizer(
            batch_size=100000,
            data_path=os.path.join(
                d_dir,
                'test.txt'),
            data_format='vowpal_wabbit',
            target_folder=target_folder_test)

        os.rename(
            os.path.join(
                target_folder_train,
                'aaaaaa.batch'),
            os.path.join(
                target_folder_train,
                'train.batch'))

        os.rename(
            os.path.join(
                target_folder_test,
                'aaaaaa.batch'),
            os.path.join(
                target_folder_test,
                'test.batch'))

        folder_for_dict = os.path.join(d_dir, 'for_dict')
        os.mkdir(folder_for_dict)

        shutil.copy(
            os.path.join(
                target_folder_train,
                'train.batch'),
            folder_for_dict)

        shutil.copy(
            os.path.join(
                target_folder_test,
                'test.batch'),
            folder_for_dict)

    def build_model(self, d_dir, n_document_passes=1):
        batch_vectorizer_train = artm.BatchVectorizer(
            data_path=os.path.join(
                d_dir,
                'data_batches_train'),
            data_format="batches")

        batch_vectorizer_test = artm.BatchVectorizer(
            data_path=os.path.join(
                d_dir,
                'data_batches_test'),
            data_format="batches")

        dictionary = artm.Dictionary()
        dictionary.gather(data_path=os.path.join(d_dir, 'for_dict'))

        model = artm.ARTM(
            num_topics=self.n_topics,
            dictionary=dictionary,
            cache_theta=True,
            reuse_theta=True)

        # Sparsity p(c|t)
        model.scores.add(
            artm.SparsityPhiScore(
                eps=EPS,
                name='SparsityPhiScoreC',
                class_id=self.c))

        # Sparsity p(w|t)
        model.scores.add(
            artm.SparsityPhiScore(
                eps=EPS,
                name='SparsityPhiScoreGram3',
                class_id=self.gram3))

        #Regularization of sparsity p(gram3|t)
        model.regularizers.add(
            artm.SmoothSparsePhiRegularizer(
                name='SparsePhiGram3Regularizer',
                class_ids=[self.gram3]))

        #Regularization of decorr p(gram3|t)
        model.regularizers.add(
           artm.DecorrelatorPhiRegularizer(
                name='DecorrelatorPhiGram3Regularizer',
                class_ids=[self.gram3]))

        model.num_document_passes = n_document_passes
        return (model,
                batch_vectorizer_train,
                batch_vectorizer_test)
    
    def process_model(self, model, batch_train, y_train, batch_test, y_test, d_dir):
        phi = initialize_phi(
                          self.init_methods,
                          self.n_topics,
                          self.n_topics_health,
                          os.path.join(d_dir, 'train.npy'),
                          os.path.join(d_dir, 'true_labels_train.npy')
                         )
        
        topics = ['topic{}'.format(t) for t in range(self.n_topics)]
        phi_new = pd.DataFrame(phi, columns=topics)
        phi_new['w'] = list(self.trigrams) + ['label0', 'label1']

        (_, phi_ref) = model.master.attach_model(model=model.model_pwt)
        model_phi = model.get_phi(model_name=model.model_pwt)

        for i, w in enumerate(model_phi.index):
            for j, t in enumerate(topics):
                phi_ref[i, j] = phi_new[phi_new.w == w][t].values[0]

        fold_scores = {'AUC(Train)':[], 'AUC(Test)':[], 'LogLoss(Train)':[], 'LogLoss(Test)':[],
                       'SparsityPhiC':[], 'SparsityPhiGram3':[]}

        gram3 = self.gram3
        c = self.c
        n_iters = self.n_iters
        n_objs = self.n_objs
        w_gram3 = 1
        tau = self.tau
        tau_phi_gram3 =  self.tau_phi_gram3

        #tau_arr = [tau] * n_iters
        tau_arr = [tau] * 5 + [tau * 10] * 5 + [tau * 100] * (n_iters - 10)
        tau_phi_gram3_arr = [0] * 5 + [tau_phi_gram3] * 5 + [tau_phi_gram3 * 5] * (n_iters - 10)
        tau_phi_gram3_decorr_arr = [self.tau_phi_gram3_decorr] * (n_iters)

        for n_iter in range(n_iters):
            model.regularizers['SparsePhiGram3Regularizer'].tau = tau_phi_gram3_arr[n_iter]
            model.regularizers['DecorrelatorPhiGram3Regularizer'].tau = tau_phi_gram3_decorr_arr[n_iter]
            model.class_ids = {
                gram3: w_gram3,
                c : tau_arr[n_iter]}

            model.fit_offline(
                num_collection_passes=1,
                batch_vectorizer=batch_train)
                        
            train_theta = model.transform(
                batch_vectorizer=batch_train,
                predict_class_id='labels').T
            y_pred_train = train_theta['label1'].values

            test_theta = model.transform(
                batch_vectorizer=batch_test,
                predict_class_id='labels').T
            y_pred_test = test_theta['label1'].values
                        
            logloss_test = log_loss(y_test, y_pred_test)
            if any(np.isnan(y_pred_train)) or any(np.isinf(y_pred_train)) or np.isnan(logloss_test) or np.isinf(logloss_test):
                break
            fold_scores['AUC(Train)'].append(roc_auc_score(y_train, y_pred_train))
            fold_scores['AUC(Test)'].append(roc_auc_score(y_test, y_pred_test))
            fold_scores['LogLoss(Train)'].append(log_loss(y_train, y_pred_train))
            fold_scores['LogLoss(Test)'].append(logloss_test)
            fold_scores['SparsityPhiC'].append(model.score_tracker['SparsityPhiScoreC'].last_value)
            fold_scores['SparsityPhiGram3'].append(model.score_tracker['SparsityPhiScoreGram3'].last_value)
            
            # p(t|c)
            theta = model.get_theta()
            p_d = 1.0 / (n_objs - y_pred_test.shape[0])
            p_t = theta
            p_t = p_t.multiply(p_d)
            p_t = p_t.sum(axis=1)

            phi = model.get_phi().reset_index()
            p_ct = phi[(phi['index'] == 'label0') | (phi['index'] == 'label1')].set_index('index')
            p_ct = p_ct.multiply(p_t)
            p_tc = p_ct.div(p_ct.sum(axis=1), axis='index').T
        
        if len(fold_scores['AUC(Train)']) == 0:
            return {'AUC(Train)':[0.0], 'AUC(Test)':[0.0], 'LogLoss(Train)':[+np.inf], 'LogLoss(Test)':[+np.inf],
                   'SparsityPhiC':[0.0], 'SparsityPhiGram3':[0.0]}, {'label0':0.0, 'label1':0.0}

        return fold_scores, p_tc

    def cross_val_score(self, n_iters=25, shuffles=[0], n_folds=10): 
        print ('n_topics:{}'.format(self.n_topics))
        print ('n_topics_health:{}'.format(self.n_topics_health))
        print('init_methods:{}'.format(self.init_methods))
        print('tau:{}'.format(self.tau))
        print('tau_phi_gram3:{}'.format(self.tau_phi_gram3))
        print('tau_phi_gram3_decorr:{}'.format(self.tau_phi_gram3_decorr))
        process_dir = self.process_dir
        self.shuffles = shuffles
        self.n_folds = n_folds
        self.n_iters = n_iters
        ptc = {'label0': np.zeros(self.n_topics), 'label1' : np.zeros(self.n_topics)}
        scores_shuffle = {}
        scores = {'AUC(Train)':{}, 'AUC(Test)':{}, 'LogLoss(Train)':{}, 'LogLoss(Test)':{}, 'SparsityPhiC':{}, 'SparsityPhiGram3':{}}
        for n_iter in range(n_iters):
            for key in scores.keys():
                scores[key][n_iter] = []
        for n_shuffle in shuffles:
            scores_shuffle[n_shuffle] = []
            print (n_shuffle)

            if os.path.isdir(process_dir):
                shutil.rmtree(process_dir)
            os.mkdir(process_dir)
            self.build_cv_folds(n_shuffle)

            for n_fold in range(n_folds):
                d_dir = '{}/data{}'.format(process_dir, n_fold)

                with open(os.path.join(d_dir, 'true_labels_train.txt'), 'r') as fin:
                    y_train = np.array([int(i) for i in fin.readlines()], dtype=int)

                with open(os.path.join(d_dir, 'true_labels_test.txt'), 'r') as fin:
                    y_test = np.array([int(i) for i in fin.readlines()], dtype=int)

                self.create_batches(d_dir)
                model, batch_train, batch_test = self.build_model(d_dir)

                fold_scores, fold_ptc  = self.process_model(
                                            model,
                                            batch_train,
                                            y_train,
                                            batch_test,
                                            y_test,
                                            d_dir
                                        )
                
                for n_iter in range(len(fold_scores['AUC(Test)'])):
                    scores['AUC(Train)'][n_iter].append(fold_scores['AUC(Train)'][n_iter])
                    scores['AUC(Test)'][n_iter].append(fold_scores['AUC(Test)'][n_iter])
                    scores['LogLoss(Train)'][n_iter].append(fold_scores['LogLoss(Train)'][n_iter])
                    scores['LogLoss(Test)'][n_iter].append(fold_scores['LogLoss(Test)'][n_iter])
                    scores['SparsityPhiC'][n_iter].append(fold_scores['SparsityPhiC'][n_iter])
                    scores['SparsityPhiGram3'][n_iter].append(fold_scores['SparsityPhiGram3'][n_iter])

                    scores_shuffle[n_shuffle].append(fold_scores['AUC(Test)'][n_iter])

                ptc['label0'] += fold_ptc['label0']
                ptc['label1'] += fold_ptc['label1']
            shutil.rmtree(process_dir)
            

        ptc['label0'] /= n_folds * len(shuffles)
        ptc['label1'] /= n_folds * len(shuffles)
        
        return scores, ptc, np.argsort(([np.mean(scores_shuffle[n_shuffle]) for n_shuffle in sorted(scores_shuffle.keys())]))[:3]

    def is_improved(self, auc_curr, logloss_curr, auc_best, logloss_best):
        if round(auc_curr, 4) > round(auc_best, 4) and round(logloss_curr, 4) < round(logloss_best, 4):
            return True

        if round(auc_curr, 4) > round(auc_best, 4) and logloss_curr < self.logloss_thr and \
           10 * (auc_curr - auc_best) > logloss_curr - logloss_best:
           return True

        if auc_curr > self.auc_thr  and round(logloss_curr, 4) < round(logloss_best, 4) and \
           10 * (auc_best - auc_curr) < logloss_best - logloss_curr:
           return True

        return False

    def find_best_params(self, search_params, auc_thr=0.8, logloss_thr=0.6, n_iters=5, n_shuffles=3):        
        self.auc_thr = auc_thr
        self.logloss_thr = logloss_thr
        
        self.tau = search_params['tau'][0]
        self.n_topics = search_params['n_topics'][0]
        self.n_topics_health = max(1, min(2, self.n_topics - 2))
        self.init_methods = [search_params['init_methods'][0]]
        self.tau_phi_gram3 = 0
        self.tau_phi_gram3_decorr = 0

        _, __, worst_shuffles = self.cross_val_score(n_iters, np.arange(n_shuffles))
        shuffles = worst_shuffles
        print('shuffles:{}'.format(shuffles))
  
        print ('n_topics')
        auc_best_global = 0
        logloss_best_global = +np.inf
        n_iters = 5
        best_inits_global = []
        for n_topics in search_params['n_topics']:
            self.n_topics = n_topics
            self.n_topics_health = max(1, min(2, n_topics - 2))
            best_inits = []
            auc_best = 0
            logloss_best = +np.inf
            while True:
                need_to_continue = False
                curr_inits  = np.setdiff1d(search_params['init_methods'], best_inits)
                if np.size(curr_inits) == 0:
                    break
                for init in curr_inits:
                    self.init_methods = best_inits + [init]
                    scores, _, __ = self.cross_val_score(n_iters, [shuffles[0]])
                    logloss_test = scores['LogLoss(Test)'][n_iters - 1]
                    if len(logloss_test) == 0:
                        continue
                    auc_curr = np.mean(scores['AUC(Test)'][n_iters - 1])
                    logloss_curr = np.mean(logloss_test)

                    print ('AUC:{}'.format(auc_curr))
                    print ('LogLoss:{}'.format(logloss_curr))
                    print ('--------------------------------------------')

                    if self.is_improved(auc_curr, logloss_curr, auc_best, logloss_best):
                        best_init = init
                        auc_best = auc_curr
                        logloss_best = logloss_curr
                        need_to_continue = True

                if not need_to_continue:
                    break
                best_inits += [best_init]

            if len(best_inits) == 0:
                continue

            if self.is_improved(auc_best, logloss_best, auc_best_global, logloss_best_global):
                global_best_inits = [init for init in best_inits]
                ntopics_best = n_topics
                auc_best_global = auc_best
                logloss_best_global = logloss_best
            
        self.n_topics = ntopics_best
        self.n_topics_health = max(1, min(2, ntopics_best - 2))
        print ('set_n_topics:{}'.format(ntopics_best))
        print ('--------------------------------------------')
        self.init_methods = [init for init in global_best_inits]
        print ('set_init_methods:{}'.format(global_best_inits))
        print ('--------------------------------------------')
        
        print ('tau')
        auc_best_global = 0
        logloss_best_global = +np.inf
        n_iters = 15
        reach_desired_sparsity = False
        for tau in search_params['tau']:
            self.tau = tau
            scores, _, __ = self.cross_val_score(n_iters, [shuffles[1]])
            logloss_test = scores['LogLoss(Test)'][n_iters - 1]
            if len(logloss_test) == 0:
                continue
            sparsity_phi_c = round(np.mean(scores['SparsityPhiC'][n_iters - 1]), 1)
            auc_curr = np.mean(scores['AUC(Test)'][n_iters - 1])
            logloss_curr = np.mean(logloss_test)

            print ('Sparsity_c:{}'.format(sparsity_phi_c))
            print ('AUC:{}'.format(auc_curr))
            print ('LogLoss:{}'.format(logloss_curr))
            print ('----------------------------------------')


            if sparsity_phi_c >= 0.5 and (auc_curr > auc_thr) and (logloss_curr < logloss_thr):
                if not reach_desired_sparsity or self.is_improved(auc_curr, logloss_curr, auc_best_global, logloss_best_global):
                    tau_best = tau
                    auc_best_global = auc_curr
                    logloss_best_global = logloss_curr
                    reach_desired_sparsity = True
 
        self.tau = tau_best
        print ('set_tau:{}'.format(tau_best))
        print ('--------------------------------------------')

        print ('tau_phi_gram3 & tau_phi_gram3_decorr')
        auc_best_global = 0
        logloss_best_global = +np.inf
        n_iters = 15
        reach_desired_sparsity = False
        for tau_phi_gram3 in search_params['tau_phi_gram3']:
            self.tau_phi_gram3 = tau_phi_gram3
            for tau_phi_gram3_decorr in search_params['tau_phi_gram3_decorr']:
                self.tau_phi_gram3_decorr = tau_phi_gram3_decorr

                scores, _, __ = self.cross_val_score(n_iters, [shuffles[2]])
                logloss_test = scores['LogLoss(Test)'][n_iters - 1]
                if len(logloss_test) == 0:
                    continue

                sparsity_phi_gram3 = round(np.mean(scores['SparsityPhiGram3'][n_iters - 1]), 1)
                auc_curr = np.mean(scores['AUC(Test)'][n_iters - 1])
                logloss_curr = np.mean(logloss_test)

                print ('Sparsity_gram3:{}'.format(sparsity_phi_gram3))
                print ('AUC:{}'.format(auc_curr))
                print ('LogLoss:{}'.format(logloss_curr))
                print ('--------------------------------------------')

                if sparsity_phi_gram3 >= 0.7 and sparsity_phi_gram3 < 1 and (auc_curr > auc_thr) and (logloss_curr < logloss_thr):
                    if not reach_desired_sparsity or self.is_improved(auc_curr, logloss_curr, auc_best_global, logloss_best_global):
                        tau_phi_gram3_best = tau_phi_gram3
                        tau_phi_gram3_decorr_best = tau_phi_gram3_decorr
                        auc_best_global = auc_curr
                        logloss_best_global = logloss_curr
                        reach_desired_sparsity = True

        self.tau_phi_gram3 = tau_phi_gram3_best 
        print ('set_tau_phi_gram3:{}'.format(tau_phi_gram3_best))
        print ('--------------------------------------------')
        self.tau_phi_gram3_decorr = tau_phi_gram3_decorr_best
        print ('set_tau_phi_gram_decorr:{}'.format(tau_phi_gram3_decorr_best))
        print ('--------------------------------------------')
        
    def valid_score(self, true_labels_train, true_labels_valid
                    ):
        data_dir = self.data_dir
        with open(os.path.join(data_dir, true_labels_train), 'r') as fin:
            y_train = np.array([int(i) for i in fin.readlines()], dtype=int)

        with open(os.path.join(data_dir, true_labels_valid), 'r') as fin:
            y_valid = np.array([int(i) for i in fin.readlines()], dtype=int)

        c = 'labels'
        gram3 = '@default_class'

        process_dir = self.process_dir
        if not os.path.isdir(process_dir):
            os.mkdir(process_dir)
        
        shutil.copy('{}/true_labels_train.txt'.format(data_dir), '{}/true_labels_train.txt'.format(process_dir))
        shutil.copy('{}/true_labels_valid.txt'.format(data_dir), '{}/true_labels_test.txt'.format(process_dir))
        shutil.copy('{}/vw_train.txt'.format(data_dir), '{}/train.txt'.format(process_dir))
        shutil.copy('{}/vw_valid.txt'.format(data_dir), '{}/test.txt'.format(process_dir))
        shutil.copy('{}/vw_train.npy'.format(data_dir), '{}/train.npy'.format(process_dir))
        shutil.copy('{}/true_labels_train.npy'.format(data_dir), '{}/true_labels_train.npy'.format(process_dir))
               

        self.create_batches(process_dir)
        model, batch_train, batch_valid = self.build_model(process_dir)

        scores, ptc  = self.process_model(
                                    model,
                                    batch_train,
                                    y_train,
                                    batch_valid,
                                    y_valid,
                                    process_dir
                                ) 

        shutil.rmtree(process_dir)
        return scores

