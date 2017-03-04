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

    def cv_folds(self, n_shuffle):
        with open('true_labels_train{}.txt'.format(n_shuffle), 'r') as fin:
            y_true = np.array([int(i) for i in fin.readlines()], dtype=str)

        with open('vw_train{}.txt'.format(n_shuffle), 'r') as fin:
            data = np.array(fin.readlines())

        skf = StratifiedKFold(y_true, self.n_folds, random_state=241)

        data_dir = self.data_dir
        for n_fold, (i, j) in enumerate(skf):

            d_dir = '{}/data{}'.format(data_dir, n_fold)
            os.mkdir(d_dir)
            with open(os.path.join(d_dir, 'test.txt'), 'w') as test:
                with open(os.path.join(d_dir, 'train.txt'), 'w') as train:
                    with open(os.path.join(d_dir, 'true_labels_test.txt'), 'w') as test_labels:
                        with open(os.path.join(d_dir, 'true_labels_train.txt'), 'w') as train_labels:

                            train.write(''.join(data[i]))
                            train_labels.write('\n'.join(y_true[i]))

                            test_labels.write('\n'.join(y_true[j]))
                            for line in data[j]:
                                test.write("%s\n" % (line.partition('|')[0]))


    def create_batches(self, d_dir):
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

    def build_model(self, d_dir, c, gram3):
        
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
            num_topics=self.n_topics,
            dictionary=dictionary,
            cache_theta=True,
            reuse_theta=True)

        # Perplexity
        #model.scores.add(artm.PerplexityScore(name='PerplexityScoreC',
        #                                      class_ids=[c],
        #                                      dictionary=dictionary))

        #model.scores.add(artm.PerplexityScore(name='PerplexityScoreGram3',
        #                                      class_ids=[gram3],
        #                                      dictionary=dictionary))

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

        model.num_document_passes = self.n_document_passes
        return (model, 
                batch_vectorizer_fit,
                batch_vectorizer_test)
    
    def process_model(self, 
                      model,
                      c, gram3,
                      tau,
                      w_gram3, 
                      tau_phi_gram3,
                      batch_vectorizer_fit,
                      y_train,
                      batch_vectorizer_test,
                      y_test):
        auc_train = []
        auc_test = []
        logloss_train = []
        logloss_test = []
        #perplexity_c = []
        #perplexity_gram3 = []
        sparsity_phi_c = []
        sparsity_phi_gram3 = []
        
        for n_iter in range(self.n_collection_passes):
            model.regularizers['SparsePhiGram3Regularizer'].tau = tau_phi_gram3[n_iter]
            model.class_ids = {
                gram3: w_gram3[n_iter],
                c: tau[n_iter]}

            model.fit_offline(
                num_collection_passes=1,
                batch_vectorizer=batch_vectorizer_fit)

            #perplexity_c.append(
            #    model.score_tracker['PerplexityScoreC'].last_value)

            #perplexity_gram3.append(
            #    model.score_tracker['PerplexityScoreGram3'].last_value)

            sparsity_phi_c.append(
                model.score_tracker['SparsityPhiScoreC'].last_value)

            sparsity_phi_gram3.append(
                model.score_tracker['SparsityPhiScoreGram3'].last_value)
            
            train_theta = model.transform(
                batch_vectorizer=batch_vectorizer_fit,
                predict_class_id='labels').T
            y_pred_train = train_theta['label1'].values
            
            test_theta = model.transform(
                batch_vectorizer=batch_vectorizer_test,
                predict_class_id='labels').T
            y_pred_test = test_theta['label1'].values

            auc_train.append(roc_auc_score(y_train, y_pred_train))
            auc_test.append(roc_auc_score(y_test, y_pred_test))

            logloss_train.append(log_loss(y_train, y_pred_train))
            logloss_test.append(log_loss(y_test, y_pred_test))
            
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
                #perplexity_c,
                #perplexity_gram3,
                sparsity_phi_c,
                sparsity_phi_gram3,
                p_tc)
               
        
    def cross_val_score(self, tau, w_gram3, tau_phi_gram3):

        data_dir = self.data_dir
        n_folds = self.n_folds
        n_topics = self.n_topics
        n_shuffles = self.n_shuffles

        c = 'labels'
        gram3 = '@default_class'

        auc_folds_iter  = {}
        logloss_folds_iter = {}
        #perplexity_c_folds_iter = {}
        #perplexity_gram3_folds_iter = {}
        sparsity_phi_c_folds_iter = {}
        sparsity_phi_gram3_folds_iter = {}

        ptc = {}
        ptc['label0'] = np.zeros(n_topics)
        ptc['label1'] = np.zeros(n_topics)

        for i in range(self.n_collection_passes):
            auc_folds_iter[i] = []
            logloss_folds_iter[i] = []
            #perplexity_c_folds_iter[i] = []
            #perplexity_gram3_folds_iter[i] = []
            sparsity_phi_c_folds_iter[i] = []
            sparsity_phi_gram3_folds_iter[i] = []

        for n_shuffle in range(n_shuffles):
            
            tl.push(n_shuffle)
            os.mkdir(data_dir)
            self.cv_folds(n_shuffle)

            for n_fold in range(n_folds):
                d_dir = '{}/data{}'.format(data_dir, n_fold)

                with open(os.path.join(d_dir, 'true_labels_train.txt'), 'r') as fin:
                    y_train = np.array([int(i) for i in fin.readlines()], dtype=int)
                    
                with open(os.path.join(d_dir, 'true_labels_test.txt'), 'r') as fin:
                    y_test = np.array([int(i) for i in fin.readlines()], dtype=int)

                self.create_batches(d_dir)
                model, batch_vectorizer_fit, batch_vectorizer_test = self.build_model(d_dir, c, gram3)
                
                (_, auc, 
                 _, logloss, 
                 #perplexity_c,
                 #perplexity_gram3,
                 sparsity_phi_c,
                 sparsity_phi_gram3,
                 p_tc) =           self.process_model(
                                            model,
                                            c, gram3,
                                            tau,
                                            w_gram3,
                                            tau_phi_gram3,
                                            batch_vectorizer_fit,
                                            y_train,
                                            batch_vectorizer_test, 
                                            y_test)
                    
                for n_iter in range(self.n_collection_passes):
                    auc_folds_iter[n_iter].append(auc[n_iter])
                    logloss_folds_iter[n_iter].append(logloss[n_iter])
                    #perplexity_c_folds_iter[n_iter].append(perplexity_c[n_iter])
                    #perplexity_gram3_folds_iter[n_iter].append(perplexity_gram3[n_iter])
                    sparsity_phi_c_folds_iter[n_iter].append(sparsity_phi_c[n_iter])
                    sparsity_phi_gram3_folds_iter[n_iter].append(sparsity_phi_gram3[n_iter])
                    
                ptc['label0'] += p_tc['label0']
                ptc['label1'] += p_tc['label1']
                
            shutil.rmtree(data_dir)

        ptc['label0'] /= n_folds * n_shuffles
        ptc['label1'] /= n_folds * n_shuffles
       
        return (auc_folds_iter, 
                logloss_folds_iter, 
                #perplexity_c_folds_iter, 
                #perplexity_gram3_folds_iter, 
                sparsity_phi_c_folds_iter, 
                sparsity_phi_gram3_folds_iter, 
                ptc)
    
    def valid_score(self,
                    data_dir,
                    true_labels_train,
                    true_labels_valid,
                    tau,
                    w_gram3, 
                    tau_phi_gram3):

        with open(os.path.join(data_dir, true_labels_train), 'r') as fin:
            y_train = np.array([int(i) for i in fin.readlines()], dtype=int)

        with open(os.path.join(data_dir, true_labels_valid), 'r') as fin:
            y_valid = np.array([int(i) for i in fin.readlines()], dtype=int)
           
        c = 'labels'
        gram3 = '@default_class'
        
        self.create_batches(data_dir)
        model, batch_vectorizer_fit, batch_vectorizer_valid = self.build_model(data_dir, c, gram3)


        return self.process_model(
                model,
                c, gram3, 
                tau,
                w_gram3,
                tau_phi_gram3,
                batch_vectorizer_fit,
                y_train, 
                batch_vectorizer_valid,
                y_valid)
    