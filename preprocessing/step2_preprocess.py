# -*- coding: utf-8 -*-
import os
import spacy
import pickle
import re


##################
# Class for corpus
##################
class Corpus(object):

    def __init__(self, dict_dataset):

        # define tokenizer
        self.tokenizer = spacy.load('en_core_web_lg', disable=['tagger', 'parser', 'ner'])
        # raw dataset
        self.dict_dataset_raw = dict_dataset
        # tokenized
        self.dict_dataset_tokenized = {}

    ####################
    # doc pre-processing
    ####################
    def preprocess_doc(self, doc):

        doc = re.sub('\"', '\'', doc)
        doc = re.sub('\'\'+', ' \' \' ', doc)
        doc = re.sub('--+', ' -- ', doc)
        doc = re.sub('\.\.+', ' .. ', doc)
        doc = re.sub('!!+', ' !! ', doc)
        doc = re.sub(',,+', ' ,, ', doc)
        doc = re.sub(',\'', ', \'', doc)
        doc = re.sub(',~~+', ' ~~ ', doc)
        doc = re.sub('/\\/\\+', ' /\\/\\ ', doc)
        doc = re.sub("((.)\\2{2})\\2+", r"\1", doc)
        doc = re.sub(r"([a-zA-Z])([!?\-:,])([a-zA-Z])", r"\1 \2 \3", doc)
        doc = re.sub(r"([a-zA-Z])([!?.\-:,\(\)])", r"\1 \2", doc)
        doc = re.sub(r"([!?.\-:,\(\)])([a-zA-Z])", r"\1 \2", doc)

        # tokenize doc
        tokens = list(self.tokenizer(doc))

        doc_new = ""
        for token in tokens:
            doc_new += token.text + " "

        return doc_new.strip()

    ################
    # split data set
    ################
    def parse_dictionary(self):

        # authors
        for a in self.dict_dataset_raw.keys():
            # fandom categories
            for f in self.dict_dataset_raw[a].keys():
                # documents
                for i, doc in enumerate(self.dict_dataset_raw[a][f]):
                    doc_tokenized = self.preprocess_doc(doc)
                    # remove short broken documents
                    if len(doc_tokenized.split()) >= 100:
                        if a not in self.dict_dataset_tokenized:
                            self.dict_dataset_tokenized[a] = {}
                        if f not in self.dict_dataset_tokenized[a]:
                            self.dict_dataset_tokenized[a][f] = []
                        self.dict_dataset_tokenized[a][f].append(doc_tokenized)


####################
# data folders/files
####################
dir_results = os.path.join('..', 'data_preprocessed')
file_results = os.path.join(dir_results, 'results.txt')

datasets = ['dict_author_fandom_doc_train',
            'dict_author_fandom_doc_cal',
            'dict_author_fandom_doc_val',
            ]

for dataset in datasets:

    open(file_results, 'a').write('preprocess ' + dataset + '...\n')

    ######
    # load
    ######
    with open(os.path.join(dir_results, dataset), 'rb') as f:
        dict_dataset = pickle.load(f)

    ############
    # preprocess
    ############
    corpus = Corpus(dict_dataset=dict_dataset)
    corpus.parse_dictionary()

    #######
    # store
    #######
    with open(os.path.join(dir_results, dataset + "_tokenized"), 'wb') as f:
        pickle.dump(corpus.dict_dataset_tokenized, f)
    del corpus

open(file_results, 'a').write('finished!' + '\n')
