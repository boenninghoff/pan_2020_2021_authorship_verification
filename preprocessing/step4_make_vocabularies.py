# -*- coding: utf-8 -*-
import fasttext
import numpy as np
import os
import pickle


##################
# Class for corpus
##################
class Corpus(object):

    def __init__(self, dict_token_counts, dict_chr_counts):

        # dictionary to count tokens/fandoms/characters
        self.dict_token_counts = dict_token_counts
        self.dict_chr_counts = dict_chr_counts

        # define token-based vocabulary
        self.V_w = {'<PAD>': 0,  # zero-padding token
                    '<UNK>': 1,  # unknown-token
                    }
        # define character-based vocabulary
        self.V_c = {'<PAD>': 0,  # zero-padding character
                    '<UNK>': 1,  # unknown-character
                    }

    ###################################
    # remove rare tokens and characters
    ###################################
    def remove_rare_tok(self, vocab_size_token):

        # remove rare token types
        q = sorted(self.dict_token_counts.items(), key=lambda x: x[1], reverse=True)
        list_tokens = list(list(zip(*q))[0])[:vocab_size_token]

        return list_tokens

    ###################################
    # remove rare tokens and characters
    ###################################
    def remove_rare_chr(self, vocab_size_chr):

        # remove rare character types
        q = sorted(self.dict_chr_counts.items(), key=lambda x: x[1], reverse=True)
        list_chr = list(list(zip(*q))[0])[:vocab_size_chr]

        return list_chr

    ############################
    # make word-based vocabulary
    ############################
    def make_wrd_vocabulary(self, list_tokens, WE_dic, D_w):

        # add tokens to vocabulary and assign an integer
        V_w = self.V_w.copy()
        for token in list_tokens:
            V_w[token] = len(V_w)

        # initialize embedding matrix
        E_w = np.zeros(shape=(len(V_w), D_w), dtype='float32')

        # fill embedding matrix
        for token in V_w.keys():
            idx = V_w[token]
            E_w[idx, :] = WE_dic[token]

        return V_w, E_w

    ############################
    # make word-based vocabulary
    ############################
    def make_chr_vocabulary(self, list_chr):

        # character vocabulary
        V_c = self.V_c.copy()
        for c in list_chr:
            V_c[c] = len(V_c)

        return V_c


############################
# define word embedding size
############################
D_w = 300

####################
# data folders/files
####################
dir_results = os.path.join('..', 'data_preprocessed')
file_results = os.path.join(dir_results, 'results.txt')

################################################
# load pre-trained fastText word embedding model
################################################
dir_WE = os.path.join('..', 'data_original', 'cc.en.300.bin')
WE_dic = fasttext.load_model(dir_WE)

#############
# load counts
#############
with open(os.path.join(dir_results, "counts_chr"), 'rb') as f:
    dict_chr_counts = pickle.load(f)
with open(os.path.join(dir_results, "counts_wrd"), 'rb') as f:
    dict_token_counts = pickle.load(f)

##################################
# prepare list of vocabulary sizes
##################################
list_vocab_size_token = [1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
list_vocab_size_chr = [100, 150, 200, 250, 300, 350, 400, 450, 500, 1000, 2000]

#############
# load corpus
#############
corpus = Corpus(dict_token_counts=dict_token_counts,
                dict_chr_counts=dict_chr_counts,
                )

for size in list_vocab_size_token:

    # remove rare tokens/characters
    list_tokens = corpus.remove_rare_tok(size)

    # create dictionaries/table
    V_w, E_w = corpus.make_wrd_vocabulary(list_tokens, WE_dic, D_w)

    # store
    with open(os.path.join(dir_results, 'vocab_wrd_' + str(size) + "_" + str(D_w)), 'wb') as f:
        pickle.dump((V_w, E_w), f)
    # print
    open(file_results, 'a').write('# wrd embeddings...' + str(len(V_w)) + ", D_w=" + str(D_w) + "\n")


for size in list_vocab_size_chr:

    # remove rare tokens/characters
    list_chr = corpus.remove_rare_chr(size)
    # create dictionaries/table
    V_c = corpus.make_chr_vocabulary(list_chr)

    # store
    with open(os.path.join(dir_results, 'vocab_chr_' + str(size)), 'wb') as f:
        pickle.dump(V_c, f)
    # print
    open(file_results, 'a').write('# chr embeddings...' + str(len(V_c)) + "\n")


open(file_results, 'a').write('finished!' + '\n')
