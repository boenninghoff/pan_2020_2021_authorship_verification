# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.join('..', 'helper_functions'))
from resample_pairs import sample_pairs_single_epoch
import numpy as np
import pickle
import os


#################################
# sample new pairs for all epochs
#################################
def resample_pairs_train(file_results):

    with open(os.path.join("..", "data_preprocessed",  'dict_author_fandom_doc_train_tokenized'), 'rb') as f:
        dict_author_fandom_doc = pickle.load(f)

    ########################################
    # sample fixed pairs for development set
    ########################################
    docs_L, docs_R, labels_a, labels_c = sample_pairs_single_epoch(dict_author_fandom_doc,
                                                                   delta_1=0.7,
                                                                   delta_2=0.6,
                                                                   delta_3=0.6,
                                                                   only_SADF_and_DASF=False,
                                                                   make_balanced=False,
                                                                   balance_factor=1.0,
                                                                   )

    #######
    # check
    #######
    # counts
    dict_counts = {"SA_SC": 0,
                   "SA_DC": 0,
                   "DA_SC": 0,
                   "DA_DC": 0,
                   }
    for i in range(len(docs_L)):

        if labels_a[i] == 1 and labels_c[i] == 1:
            dict_counts["SA_SC"] += 1
        if labels_a[i] == 1 and labels_c[i] == 0:
            dict_counts["SA_DC"] += 1
        if labels_a[i] == 0 and labels_c[i] == 1:
            dict_counts["DA_SC"] += 1
        if labels_a[i] == 0 and labels_c[i] == 0:
            dict_counts["DA_DC"] += 1

    s = "-----"
    open(file_results, 'a').write('\n' + s + '\n')

    s = '# sampled pairs (train) || a=0: ' + str(np.sum(np.array(labels_a) == 0)) \
        + ', a=1: ' + str(np.sum(np.array(labels_a) == 1)) \
        + ', c=0: ' + str(np.sum(np.array(labels_c) == 0)) \
        + ', c=1: ' + str(np.sum(np.array(labels_c) == 1))
    open(file_results, 'a').write(s + '\n')

    s = 'SA_SC: ' + str(dict_counts["SA_SC"]) \
        + ', SA_DC: ' + str(dict_counts["SA_DC"]) \
        + ', DA_SC: ' + str(dict_counts["DA_SC"]) \
        + ', DA_DC: ' + str(dict_counts["DA_DC"])
    open(file_results, 'a').write(s + '\n\n')

    return docs_L, docs_R, labels_a, labels_c


