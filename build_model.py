import shutil
import re
import os
import sys
from collections import Counter
from itertools import product
import operator
import string
import logging

from sklearn.metrics import roc_auc_score, log_loss
from sklearn.cross_validation import StratifiedKFold
import numpy as np
import pandas as pd

# Configure logging folder
import artm
os.environ["ARTM_SHARED_LIBRARY"] = "/root/bigartm/build/lib/libartm.so"

lc = artm.messages.ConfigureLoggingArgs()
lc.log_dir = r'tmp'
lib = artm.wrapper.LibArtm(logging_config=lc)

# Change any other logging parameters at runtime (except logging folder) 
lc.minloglevel=2  # 0 = INFO, 1 = WARNING, 2 = ERROR, 3 = FATAL
lib.ArtmConfigureLogging(lc)

# Telegram bot
import sys
sys.path.insert(1,'/notebooks/pyloggers')
from telegram_logger import TelegramLogger
tl = TelegramLogger(name='artm')

EPS_sparse = 1e-4
class EkgClassification(object):

    def __init__(self,
                 data_dir,
                 n_objs,
                 n_topics,
                 n_collection_passes,
                 n_document_passes,
                 n_shuffles=10,
                 n_folds=10, 
                 eps=1e-4):

        self.data_dir = data_dir
        self.n_objs = n_objs
        self.n_topics = n_topics
        self.n_collection_passes = n_collection_passes
        self.n_document_passes = n_document_passes
        self.n_shuffles = n_shuffles
        self.n_folds = n_folds
        self.eps = eps

    def reject_shuffle(self, lines, n_shuffles):
        """Shuffle lines n_shuffles times.

        Parameters:
        ===========
        lines :
           Lines to shuffle.

        n_shuffles :
           The number of times to shuffle.
        """
        orders = set()
        e_strings = list(enumerate(lines))

        for gen_idx in range(n_shuffles):

            while tuple([idx for idx, _ in e_strings]) in orders:
                  np.random.shuffle(e_strings)

            orders.update((tuple([idx for idx, _ in e_strings]),))
            yield [val for _, val in e_strings]


    def create_shuffled_vw(self):
        """Creates shuffled vw files."""
        with open('vw.txt', 'r') as fin:
            lines = fin.readlines()

        for gen_idx, shuffled in enumerate(self.reject_shuffle(lines, self.n_shuffles)):
            y_true = []
            for line in shuffled:
                line_tmp = "%s " % (line.split('|', 1)[1][:-1])
                str_tmp = re.findall(r'\:(.*?)\ ', line_tmp)
                y_true.append(str_tmp[1])

                with open('true_labels_vw{}.txt'.format(gen_idx), 'w') as fout:
                    fout.write('\n'.join(y_true))

                with open('vw{}.txt'.format(gen_idx), 'w') as fout:
                    fout.write(''.join(shuffled))


    def cv_folds(self, n_shuffle):
        """Creates folds for cross-validation.

        Parameters:
        ===========
        n_shuffle : 
           The number of shuffled vowpal-wabbit files.
        """
        # Initialization
        # docs_freq = {}
        # docs_freq3 = {}

        with open('true_labels_vw{}.txt'.format(n_shuffle), 'r') as fin:
            y_true = np.array([int(i) for i in fin.readlines()], dtype=str)
            
        with open('vw{}.txt'.format(n_shuffle), 'r') as fin:
            data = np.array(fin.readlines())
            
        skf = StratifiedKFold(y_true, self.n_folds, random_state=241)

        data_dir = self.data_dir
        for n_fold, (i, j) in enumerate(skf):

            d_dir = '{}/data{}'.format(data_dir, n_fold)
            os.mkdir(d_dir)
            with open(os.path.join(d_dir, 'test.txt'), 'w') as test:
                with open(os.path.join(d_dir, 'true_labels.txt'), 'w') as label:
                    with open(os.path.join(d_dir, 'train.txt'), 'w') as train:

                        train.write(''.join(data[i]))
                        label.write('\n'.join(y_true[j]))

                        for line in data[j]:
                            test.write("%s\n" % (line.partition('|')[0]))

        # return docs_freq, docs_freq3


    def create_batches(self, d_dir):
        """Create train and test batches in d_dir.

        Parameters:
        ===========
        d_dir : 
           The data directory.
        """

        batch_vectorize = None
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

    def build_model(self, tau, w_gram3, tau_theta, tau_phi_gram3):
        self.create_shuffled_vw()

        data_dir = self.data_dir
        n_topics = self.n_topics
        n_collection_passes = self.n_collection_passes
        n_document_passes = self.n_document_passes 
        n_shuffles = self.n_shuffles
        n_folds = self.n_folds

        c = 'labels'
        gram3 = '@default_class'

        auc_folds_iter  = {}
        logloss_folds_iter = {}
        perplexity_c = {}
        perplexity_gram3 = {}
        sparsity_phi_c = {}
        sparsity_phi_gram3 = {}
        sparsity_theta = {}

        ptc = {}
        ptc['label0'] = np.zeros(n_topics)
        ptc['label1'] = np.zeros(n_topics)

        for i in range(n_collection_passes):
            auc_folds_iter[i] = []
            logloss_folds_iter[i] = []
            perplexity_c[i] = []
            perplexity_gram3[i] = []
            sparsity_phi_c[i] = []
            sparsity_phi_gram3[i] = []
            sparsity_theta[i] = []

        for n_shuffle in range(n_shuffles):
            tl.push(n_shuffle)
            os.mkdir(data_dir)
            # docs_freq, docs_freq3 =
            self.cv_folds(n_shuffle)

            for n_fold in range(n_folds):
                # test_labels_file = os.path.join(data_dir, 'true_labels.txt')
                # true_p_cd = []
                # with open(test_labels_file, 'r') as fin:
                #    for line in fin.readlines():
                #        if int(line.split(" ")[0]):
                #            true_p_cd.append(0)
                #        else:
                #            true_p_cd.append(1)

                # true_p_cd = np.asarray(true_p_cd)

                d_dir = '{}/data{}'.format(data_dir, n_fold)

                with open(os.path.join(d_dir, 'true_labels.txt'), 'r') as fin:
                    y_true = np.array([int(i) for i in fin.readlines()], dtype=int)

                self.create_batches(d_dir)

                batch_vectorizer_fit = artm.BatchVectorizer(
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
                    num_topics=n_topics,
                    dictionary=dictionary,
                    cache_theta=True,
                    reuse_theta=True)

                # Initialization
                #(_, phi_ref) = model.master.attach_model(model=model.model_pwt)
                # phi_new = build_phi(n_topics, docs_freq, docs_freq3)

                # ind = list(model.get_phi(model_name=model.model_pwt).reset_index()['index'])
                # phi_new = phi_new.reindex(ind).as_matrix()

                # for tok in xrange(n_tokens):
                #    for top in xrange(n_topics):
                #        phi_ref[tok, top] = phi_new[tok, top]

                # Perplexity
                model.scores.add(artm.PerplexityScore(name='PerplexityScoreC',
                                                      class_ids=[c],
                                                      dictionary=dictionary))

                model.scores.add(artm.PerplexityScore(name='PerplexityScoreGram3',
                                                      class_ids=[gram3],
                                                      dictionary=dictionary))

                # Sparsity p(c|t) 
                model.scores.add(
                    artm.SparsityPhiScore(
                        eps=EPS_sparse,
                        name='SparsityPhiScoreC',
                        class_id=c))

                # Sparsity p(w|t)
                model.scores.add(
                    artm.SparsityPhiScore(
                        eps=EPS_sparse,
                        name='SparsityPhiScoreGram3',
                        class_id=gram3))
                
                # Sparsity p(t|d)
                model.scores.add(
                    artm.SparsityThetaScore(
                        eps=EPS_sparse,
                        name='SparsityThetaScore'))
                
                # Regularization of sparsity p(gram3|t)
                model.regularizers.add(
                    artm.SmoothSparsePhiRegularizer(
                        name='SparsePhiGram3Regularizer', 
                        class_ids=[gram3]))
                
                # Regularization of sparsity p(t|d)
                model.regularizers.add(
                     artm.SmoothSparseThetaRegularizer(
                       name='SparsetThetaRegularizer'))

                model.num_document_passes = n_document_passes

                for n_iter in range(n_collection_passes):
                    
                    model.regularizers['SparsePhiGram3Regularizer'].tau = tau_phi_gram3[n_iter]
                    model.regularizers['SparsetThetaRegularizer'].tau = tau_theta[n_iter]
                    
                    model.class_ids = {
                        gram3: w_gram3[n_iter],
                        c: tau[n_iter]}

                    model.fit_offline(
                        num_collection_passes=1,
                        batch_vectorizer=batch_vectorizer_fit)

                    perplexity_c[n_iter].append(
                        model.score_tracker['PerplexityScoreC'].last_value)

                    perplexity_gram3[n_iter].append(
                        model.score_tracker['PerplexityScoreGram3'].last_value)

                    sparsity_phi_c[n_iter].append(
                        model.score_tracker['SparsityPhiScoreC'].last_value)

                    sparsity_phi_gram3[n_iter].append(
                        model.score_tracker['SparsityPhiScoreGram3'].last_value)
                    
                    sparsity_theta[n_iter].append(
                        model.score_tracker['SparsityThetaScore'].last_value)

                    test_theta = model.transform(
                        batch_vectorizer=batch_vectorizer_test,
                        predict_class_id='labels').T
                    y_pred = test_theta['label1'].values

                    auc_folds_iter[n_iter].append(
                        roc_auc_score(y_true, y_pred))

                    logloss_folds_iter[n_iter].append(
                        log_loss(y_true, y_pred))

                # p(t|c)
                theta = model.get_theta()
                p_d = 1.0 / (self.n_objs - y_pred.shape[0])
                p_t = theta
                p_t = p_t.multiply(p_d)
                p_t = p_t.sum(axis=1)

                phi = model.get_phi().reset_index()
                p_ct = phi[(phi['index'] == 'label0') | (
                    phi['index'] == 'label1')].set_index('index')
                p_ct = p_ct.multiply(p_t)
                p_tc = p_ct.div(p_ct.sum(axis=1), axis='index').T

                # p_tc.to_csv(ptc_file)
                ptc['label0'] += p_tc['label0']
                ptc['label1'] += p_tc['label1']


            # ptc['label0'] /= n_folds
            # ptc['label1'] /= n_folds

            shutil.rmtree(data_dir)

        ptc['label0'] /= n_folds * n_shuffles
        ptc['label1'] /= n_folds * n_shuffles

        return (auc_folds_iter, 
                logloss_folds_iter, 
                perplexity_c, 
                perplexity_gram3, 
                sparsity_phi_c, 
                sparsity_phi_gram3, 
                sparsity_theta,
                ptc)


