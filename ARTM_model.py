import shutil
import string
from itertools import product

from sklearn.metrics import roc_auc_score, log_loss
from sklearn.cross_validation import StratifiedKFold
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
lc.log_dir = r'tmp'
lib = artm.wrapper.LibArtm(logging_config=lc)

# Change any other logging parameters at runtime (except logging folder)
lc.minloglevel=2  # 0 = INFO, 1 = WARNING, 2 = ERROR, 3 = FATAL
lib.ArtmConfigureLogging(lc)

# Telegram bot
#sys.path.insert(1,'pyloggers')
#from telegram_logger import TelegramLogger
#tl = TelegramLogger(name='artm')

class EcgClassification(object):

    def __init__(self,
                 data_dir,
                 n_objs,
                 n_topics,
                 n_topics_health,
                 n_collection_passes=25,
                 n_document_passes=1,
                 n_shuffles=10,
                 n_folds=10,
                 process_dir='data_ecg',
                 init_method='KMeans',
                 eps=1e-4):

        self.data_dir = data_dir
        self.n_objs = n_objs
        self.n_topics = n_topics
        self.n_topics_health = n_topics_health
        self.n_collection_passes = n_collection_passes
        self.n_document_passes = n_document_passes
        self.n_shuffles = n_shuffles
        self.n_folds = n_folds
        self.process_dir = process_dir
        self.init_method = init_method
        self.eps = eps

    def trigrams(self):
        tokens = []
        for token in product(string.ascii_uppercase, repeat=3):

            if token[0] > 'F':
                break

            if token[1] > 'F' or token[2] > 'F':
                continue

            tokens.append(''.join(token))
        return tokens

    def cv_folds(self, n_shuffle):
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

    def build_model(self, d_dir, c, gram3):

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
                eps=self.eps,
                name='SparsityPhiScoreC',
                class_id=c))

        # Sparsity p(w|t)
        model.scores.add(
            artm.SparsityPhiScore(
                eps=self.eps,
                name='SparsityPhiScoreGram3',
                class_id=gram3))

        # Regularization of sparsity p(gram3|t)
        model.regularizers.add(
            artm.SmoothSparsePhiRegularizer(
                name='SparsePhiGram3Regularizer',
                class_ids=[gram3]))

        #model.regularizers.add(
        #    artm.DecorrelatorPhiRegularizer(
        #        name='DecorrelatorPhiGram3Regularizer',
        #        class_ids=[gram3]))

        model.num_document_passes = self.n_document_passes
        return (model,
                batch_vectorizer_train,
                batch_vectorizer_test)
    
    def process_model(self,
                      model,
                      c,
                      gram3,
                      tau,
                      tau_phi_gram3,
                      batch_vectorizer_train,
                      y_train,
                      batch_vectorizer_test,
                      y_test,
                      d_dir):
        auc_train = []
        auc_test = []
        logloss_train = []
        logloss_test = []
        sparsity_phi_c = []
        sparsity_phi_gram3 = []

        phi = initialize_phi(
                          self.init_method,
                          self.n_topics,
                          self.n_topics_health,
                          os.path.join(d_dir, 'train.npy'),
                          os.path.join(d_dir, 'true_labels_train.npy')
                         )
        n_iters = self.n_collection_passes
        w_gram3 = [1] * n_iters
        topics = ['topic{}'.format(t) for t in range(self.n_topics)]
        phi_new = pd.DataFrame(phi, columns=topics)
        phi_new['w'] = self.trigrams() + ['label0', 'label1']

        (_, phi_ref) = model.master.attach_model(model=model.model_pwt)
        model_phi = model.get_phi(model_name=model.model_pwt)

        for i, w in enumerate(model_phi.index):
            for j, t in enumerate(topics):
                phi_ref[i, j] = phi_new[phi_new.w == w][t].values[0]
   
        for n_iter in range(n_iters):
            model.regularizers['SparsePhiGram3Regularizer'].tau = tau_phi_gram3[n_iter]
            
            model.class_ids = {
                gram3: w_gram3[n_iter],
                c: tau[n_iter]}

            model.fit_offline(
                num_collection_passes=1,
                batch_vectorizer=batch_vectorizer_train)

            sparsity_phi_c.append(
                model.score_tracker['SparsityPhiScoreC'].last_value)

            sparsity_phi_gram3.append(
                model.score_tracker['SparsityPhiScoreGram3'].last_value)

            train_theta = model.transform(
                batch_vectorizer=batch_vectorizer_train,
                predict_class_id='labels').T
            y_pred_train = train_theta['label1'].values

            test_theta = model.transform(
                batch_vectorizer=batch_vectorizer_test,
                predict_class_id='labels').T
            y_pred_test = test_theta['label1'].values
                        
            logloss = log_loss(y_test, y_pred_test)
            if np.isnan(logloss) or np.isinf(logloss):
                break
            auc_train.append(roc_auc_score(y_train, y_pred_train))
            auc_test.append(roc_auc_score(y_test, y_pred_test))
            
            logloss_train.append(log_loss(y_train, y_pred_train))
            logloss_test.append(logloss)
            
            # p(t|c)
            theta = model.get_theta()
            p_d = 1.0 / (self.n_objs - y_pred_test.shape[0])
            p_t = theta
            p_t = p_t.multiply(p_d)
            p_t = p_t.sum(axis=1)

            phi = model.get_phi().reset_index()
            p_ct = phi[(phi['index'] == 'label0') | (
                    phi['index'] == 'label1')].set_index('index')
            p_ct = p_ct.multiply(p_t)
            p_tc = p_ct.div(p_ct.sum(axis=1), axis='index').T

        return (auc_train,
                auc_test,
                logloss_train,
                logloss_test,
                sparsity_phi_c,
                sparsity_phi_gram3,
                p_tc)

    def cross_val_score(self, tau, tau_phi_gram3):

        process_dir = self.process_dir
        data_dir = self.data_dir
        n_folds = self.n_folds
        n_topics = self.n_topics
        n_shuffles = self.n_shuffles

        c = 'labels'
        gram3 = '@default_class'

        auc_folds_iter  = {}
        logloss_folds_iter = {}
        sparsity_phi_c_folds_iter = {}
        sparsity_phi_gram3_folds_iter = {}

        ptc = {}
        ptc['label0'] = np.zeros(n_topics)
        ptc['label1'] = np.zeros(n_topics)

        for i in range(self.n_collection_passes):
            auc_folds_iter[i] = []
            logloss_folds_iter[i] = []
            sparsity_phi_c_folds_iter[i] = []
            sparsity_phi_gram3_folds_iter[i] = []

        for n_shuffle in range(n_shuffles):
            if os.path.isdir(process_dir):
                shutil.rmtree(process_dir)
            os.mkdir(process_dir)
            self.cv_folds(n_shuffle)

            for n_fold in range(n_folds):
                d_dir = '{}/data{}'.format(process_dir, n_fold)

                with open(os.path.join(d_dir, 'true_labels_train.txt'), 'r') as fin:
                    y_train = np.array([int(i) for i in fin.readlines()], dtype=int)

                with open(os.path.join(d_dir, 'true_labels_test.txt'), 'r') as fin:
                    y_test = np.array([int(i) for i in fin.readlines()], dtype=int)

                self.create_batches(d_dir)
                model, batch_vectorizer_train, batch_vectorizer_test = self.build_model(d_dir, c, gram3)

                (_, auc,
                 _, logloss,
                 sparsity_phi_c,
                 sparsity_phi_gram3,
                 p_tc) =           self.process_model(
                                            model,
                                            c,
                                            gram3,
                                            tau,
                                            tau_phi_gram3,
                                            batch_vectorizer_train,
                                            y_train,
                                            batch_vectorizer_test,
                                            y_test,
                                            d_dir)

                for n_iter in range(len(auc)):
                    auc_folds_iter[n_iter].append(auc[n_iter])
                    logloss_folds_iter[n_iter].append(logloss[n_iter])
                    sparsity_phi_c_folds_iter[n_iter].append(sparsity_phi_c[n_iter])
                    sparsity_phi_gram3_folds_iter[n_iter].append(sparsity_phi_gram3[n_iter])

                ptc['label0'] += p_tc['label0']
                ptc['label1'] += p_tc['label1']

            shutil.rmtree(process_dir)
            print (n_shuffle)

        ptc['label0'] /= n_folds * n_shuffles
        ptc['label1'] /= n_folds * n_shuffles
        return (auc_folds_iter,
                logloss_folds_iter,
                sparsity_phi_c_folds_iter,
                sparsity_phi_gram3_folds_iter,
                ptc)

    def valid_score(self,
                    true_labels_train,
                    true_labels_valid,
                    tau,
                    tau_phi_gram3):

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
        shutil.copy('{}/vw_train.txt'.format(data_dir), 'train.txt'.format(process_dir))
        shutil.copy('{}/vw_valid.txt'.format(data_dir), 'test.txt'.format(process_dir))
        shutil.copy('{}/vw_train.npy'.format(data_dir), 'train.npy'.format(process_dir))
        shutil.copy('{}/true_labels_train.npy'.format(data_dir), 'true_labels_train.npy'.format(process_dir))
        self.create_batches(process_dir)
        model, batch_vectorizer_train, batch_vectorizer_valid = self.build_model(process_dir, c, gram3)
        res = self.process_model(
                model,
                c,
                gram3,
                tau,
                tau_phi_gram3,
                batch_vectorizer_train,
                y_train,
                batch_vectorizer_valid,
                y_valid,
                process_dir)

        shutil.rmtree(process_dir)
        return res
'''data_dir = 'XX'
process_dir = 'data_ekg'
if os.path.isdir(process_dir):
    shutil.rmtree(process_dir)
n_objs = 548
n_topics = 2
n_topics_health = 1
n_collection_passes = 25
n_document_passes = 1
n_shuffles = 10
n_folds = 10
clf = EcgClassification(
                      data_dir,
                      process_dir,
                      n_objs,
                      n_topics,
                      n_topics_health,
                      n_collection_passes,
                      n_document_passes,
                      n_shuffles,
                      n_folds
                      )

#2 topics
tau = np.hstack((np.repeat([1e2, 1e2 * 5], [5, 5]), np.repeat([1e3, 1e3 * 5], [5, 5]), np.repeat([1e4], [5])))
tau_phi_gram3 = np.hstack((np.hstack(([0] * 5, np.linspace(-1e2, -1e3, 15))), [-1e3] * 5))

#8 topics
#tau = np.repeat([1e2, 1e3, 1e4], [10, 8, 7])
#tau_phi_gram3 = np.hstack((np.hstack(([0] * 5, np.linspace(-1e2, -1e2, 15))), [-1e2] * 5))
w_gram3   = [1] * n_collection_passes

(auc_folds_iter,
 logloss_folds_iter,
 sparsity_phi_c_folds_iter,
 sparsity_phi_gram3_folds_iter,
 ptc) =  clf.cross_val_score(tau, w_gram3, tau_phi_gram3)'''
