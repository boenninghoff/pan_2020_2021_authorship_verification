# -*- coding: utf-8 -*-
import os
import pickle


##################
# Class for corpus
##################
class Corpus(object):

    def __init__(self, dict_author_fandom_docs):

        # dataset (tokenized)
        self.dict_author_fandom_docs = dict_author_fandom_docs

        # dictionary to count tokens/fandoms/characters
        self.dict_token_counts = {}
        self.dict_chr_counts = {}

    ####################
    # doc pre-processing
    ####################
    def count_docs(self, doc):

        tokens = doc.split()

        for token in tokens:
            self.count_tokens_and_characters(token)

    #########################################
    # count total number of tokens/characters
    #########################################
    def count_tokens_and_characters(self, token):

        for chr in token:
            if chr not in self.dict_chr_counts:
                self.dict_chr_counts[chr] = 0
            self.dict_chr_counts[chr] += 1

        if token not in self.dict_token_counts:
            self.dict_token_counts[token] = 0
        self.dict_token_counts[token] += 1

    ################
    # split data set
    ################
    def parse_dictionary(self):

        # authors
        for a in self.dict_author_fandom_docs.keys():
            # fandom categories
            for f in self.dict_author_fandom_docs[a].keys():
                # documents
                for doc in self.dict_author_fandom_docs[a][f]:
                    self.count_docs(doc)


####################
# data folders/files
####################
dir_results = os.path.join('..', 'data_preprocessed')
file_results = os.path.join(dir_results, 'results.txt')


open(file_results, 'a').write('count tokens/characters in train set...\n')

######
# load
######
with open(os.path.join(dir_results , 'dict_author_fandom_doc_train_tokenized'), 'rb') as f:
    dict_author_fandom_doc = pickle.load(f)

################
# prepare corpus
################
corpus = Corpus(dict_author_fandom_docs=dict_author_fandom_doc)

########################################
# parse docs and count tokens/characters
########################################
corpus.parse_dictionary()


#######
# store
#######
with open(os.path.join(dir_results, "counts_chr"), 'wb') as f:
    pickle.dump(corpus.dict_chr_counts, f)
with open(os.path.join(dir_results, "counts_wrd"), 'wb') as f:
    pickle.dump(corpus.dict_token_counts, f)

open(file_results, 'a').write("number of characters: " + str(len(corpus.dict_chr_counts)) + "\n")
open(file_results, 'a').write("number of tokens: " + str(len(corpus.dict_token_counts)) + "\n")

open(file_results, 'a').write('finished!' + '\n')
