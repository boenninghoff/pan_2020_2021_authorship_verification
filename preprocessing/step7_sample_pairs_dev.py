# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.join('..', 'helper_functions'))
from resample_pairs import sample_pairs_single_epoch
import numpy as np
import pickle
import os

##################################
# load calibration/validation sets
##################################
dir_results = os.path.join('..', 'data_preprocessed')
file_results = os.path.join(dir_results, 'results.txt')

open(file_results, 'a').write('-----------------------------------------------\n')

with open(os.path.join(dir_results, 'dict_author_fandom_doc_cal_tokenized'), 'rb') as f:
    dict_author_fandom_doc_cal = pickle.load(f)
with open(os.path.join(dir_results, 'dict_author_fandom_doc_val_tokenized'), 'rb') as f:
    dict_author_fandom_doc_val = pickle.load(f)
dict_author_fandom_doc = {**dict_author_fandom_doc_cal, **dict_author_fandom_doc_val}


########################################
# sample fixed pairs for development set
########################################
docs_L, docs_R, labels_a, labels_c = sample_pairs_single_epoch(dict_author_fandom_doc,
                                                               delta_1=0.9,
                                                               delta_2=0.9,
                                                               delta_3=0.7,
                                                               only_SADF_and_DASF=True,
                                                               make_balanced=True,
                                                               balance_factor=1.0,
                                                               )
###################
# check re-sampling
###################
# counts
dict_counts = {"SA_SF": 0,
               "SA_DF": 0,
               "DA_SF": 0,
               "DA_DF": 0,
               }
for i in range(len(docs_L)):

    if labels_a[i] == 1 and labels_c[i] == 1:
        dict_counts["SA_SF"] += 1
    if labels_a[i] == 1 and labels_c[i] == 0:
        dict_counts["SA_DF"] += 1
    if labels_a[i] == 0 and labels_c[i] == 1:
        dict_counts["DA_SF"] += 1
    if labels_a[i] == 0 and labels_c[i] == 0:
        dict_counts["DA_DF"] += 1

open(file_results, 'a').write('dev: '
                              + ', #pairs: ' + str(len(labels_a))
                              + ', a=0: ' + str(np.sum(np.array(labels_a) == 0))
                              + ', a=1: ' + str(np.sum(np.array(labels_a) == 1))
                              + ', c=0: ' + str(np.sum(np.array(labels_c) == 0))
                              + ', c=1: ' + str(np.sum(np.array(labels_c) == 1))
                              + '\n')
open(file_results, 'a').write('SA_SF: ' + str(dict_counts["SA_SF"])
                              + ', SA_DF: ' + str(dict_counts["SA_DF"])
                              + ', DA_SF: ' + str(dict_counts["DA_SF"])
                              + ', DA_DF: ' + str(dict_counts["DA_DF"])
                              + '\n')
open(file_results, 'a').write('-----------------------------------------------\n')

#########################
# check overlapping ratio
#########################
with open(os.path.join(dir_results, 'pairs_val'), 'rb') as f:
    docs_L_val, docs_R_val, _, _ = pickle.load(f)
with open(os.path.join(dir_results, 'pairs_cal'), 'rb') as f:
    docs_L_cal, docs_R_cal, _, _ = pickle.load(f)

docs_L_check, docs_R_check = docs_L_val + docs_L_cal, docs_R_val + docs_R_cal

counter_1 = 0
counter_0 = 0
for doc_L, doc_R in zip(docs_L, docs_R):
    if doc_L in docs_L_check:
        idx = docs_L_check.index(doc_L)
        if doc_R == docs_R_check[idx]:
            counter_1 += 1
        else:
            counter_0 += 1
    elif doc_L in docs_R_check:
        idx = docs_R_check.index(doc_L)
        if doc_R == docs_L_check[idx]:
            counter_1 += 1
        else:
            counter_0 += 1
    else:
        counter_0 += 1

open(file_results, 'a').write('# pairs in cal/val set: ' + str(len(docs_L_check)) + '\n')
open(file_results, 'a').write('# cal/val pairs in dev set: ' + str(counter_1) + '\n')
open(file_results, 'a').write('# cal/val pairs not in dev set: ' + str(counter_0) + '\n')
open(file_results, 'a').write('-----------------------------------------------\n')

#######
# store
#######
with open(os.path.join(dir_results, 'pairs_dev'), 'wb') as f:
    pickle.dump((docs_L, docs_R, labels_a, labels_c), f)
