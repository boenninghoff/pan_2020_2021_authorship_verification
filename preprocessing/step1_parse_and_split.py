# -*- coding: utf-8 -*-
import os
import pickle
import json
from sklearn.utils import shuffle
import random


#################
# parse json file
#################
def parse(path):
    g = open(path, 'r')
    for l in g:
        yield json.loads(l.strip())


##################
# Class for corpus
##################
class Corpus(object):

    def __init__(self):

        # dictionary with all documents
        self.dict_author_fandom_doc = {}

        # sets with unique authors/fandoms/docs
        self.authors = set()
        self.fandoms = set()
        self.unique_docs = set()

        # author/fandom sets for training set (train DML, BFS, UAL)
        self.authors_train = set()
        self.fandoms_train = set()
        # author/fandom sets for calibration set (train O2D2)
        self.authors_cal = set()
        self.fandoms_cal = set()
        # author/fandom sets for validation set (check results)
        self.authors_val = set()
        self.fandoms_val = set()

        # dictionaries for the splits
        self.dict_author_fandom_doc_train = {}
        self.dict_author_fandom_doc_cal = {}
        self.dict_author_fandom_doc_val = {}

        # counts
        self.n_train = 0
        self.n_cal = 0
        self.n_val = 0
        self.n_dropped = 0

    ##################
    # parse train docs
    ##################
    def parse_raw_data(self, dir_pairs, dir_labels):

        # open json files
        with open(dir_pairs, 'r') as f:
            lines_pairs = f.readlines()
        with open(dir_labels, 'r') as f:
            lines_labels = f.readlines()

        for n in range(len(lines_pairs)):

            pair, label = json.loads(lines_pairs[n].strip()), json.loads(lines_labels[n].strip())

            for i in range(2):

                # get author-ID, fandom, fanfiction
                author = label['authors'][i]
                fandom = pair['fandoms'][i]
                doc = pair['pair'][i]

                # remove "broken" or very short documents
                if doc not in self.unique_docs:

                    self.unique_docs.add(doc)
                    self.authors.add(author)
                    self.fandoms.add(fandom)

                    if author not in self.dict_author_fandom_doc.keys():
                        self.dict_author_fandom_doc[author] = {}
                    if fandom not in self.dict_author_fandom_doc[author].keys():
                        self.dict_author_fandom_doc[author][fandom] = []
                    self.dict_author_fandom_doc[author][fandom].append(doc)

    ################
    # split data set
    ################
    def split_data(self):

        #########################
        # step 0: shuffle fandoms
        #########################
        fandoms = list(self.fandoms)
        fandoms = shuffle(fandoms)

        ##################################################
        # step 1: split fandoms into three disjoint groups
        ##################################################
        # make fandom list for train set
        n = int(0.75 * len(fandoms))
        fandoms_train = fandoms[:n]
        fandoms = shuffle(fandoms[n:])

        # make fandom list for calibration/validation sets
        n = int(0.5 * len(fandoms))
        fandoms_cal = fandoms[:n]
        fandoms_val = fandoms[n:]

        ##########################################################
        # step 2: add authors to the groups (includes overlapping)
        ##########################################################
        authors_train = set()
        authors_cal = set()
        authors_val = set()

        for a in self.authors:
            for f in self.dict_author_fandom_doc[a].keys():
                if f in fandoms_train:
                    authors_train.add(a)
                if f in fandoms_cal:
                    authors_cal.add(a)
                if f in fandoms_val:
                    authors_val.add(a)

        ################################################
        # step 3: make disjoint groups w.r.t. authorship
        ################################################
        for a in self.authors:

            # make cal/val sets disjoint
            if a in authors_cal and a in authors_val:
                if random.uniform(0, 1) < 0.5:
                    authors_cal.remove(a)
                else:
                    authors_val.remove(a)

            # make train/cal and train/cal sets disjoint
            if a in authors_train:

                if a in authors_cal:
                    if random.uniform(0, 1) < 0.5:
                        authors_cal.remove(a)
                    else:
                        authors_train.remove(a)

                if a in authors_val:
                    if random.uniform(0, 1) < 0.5:
                        authors_val.remove(a)
                    else:
                        authors_train.remove(a)

        ##############################
        # step 4: prepare dictionaries
        ##############################
        for a in self.authors:
            for f in self.dict_author_fandom_doc[a]:

                #################
                # calibration set
                #################
                if a in authors_cal and f in fandoms_cal:
                    if a not in self.dict_author_fandom_doc_cal.keys():
                        self.dict_author_fandom_doc_cal[a] = {}
                    self.dict_author_fandom_doc_cal[a][f] = self.dict_author_fandom_doc[a][f]
                    self.n_cal += len(self.dict_author_fandom_doc_cal[a][f])
                    self.authors_cal.add(a)
                    self.fandoms_cal.add(f)

                ################
                # validation set
                ################
                elif a in authors_val and f in fandoms_val:
                    if a not in self.dict_author_fandom_doc_val.keys():
                        self.dict_author_fandom_doc_val[a] = {}
                    self.dict_author_fandom_doc_val[a][f] = self.dict_author_fandom_doc[a][f]
                    self.n_val += len(self.dict_author_fandom_doc_val[a][f])
                    self.authors_val.add(a)
                    self.fandoms_val.add(f)

                ###########
                # train set
                ###########
                elif a in authors_train and f in fandoms_train:
                    if a not in self.dict_author_fandom_doc_train.keys():
                        self.dict_author_fandom_doc_train[a] = {}
                    self.dict_author_fandom_doc_train[a][f] = self.dict_author_fandom_doc[a][f]
                    self.n_train += len(self.dict_author_fandom_doc_train[a][f])
                    self.authors_train.add(a)
                    self.fandoms_train.add(f)

                ###################
                # dropped documents
                ###################
                else:
                    self.n_dropped += len(self.dict_author_fandom_doc[a][f])


########################################################################
# original large data set (expected to be in the folder "data_original")
########################################################################
dir_pairs_PAN = os.path.join('..', 'data_original', 'pairs.jsonl')
dir_label_PAN = os.path.join('..', 'data_original', 'labels.jsonl')

#####################################
# create folder for preprocessed data
#####################################
dir_results = os.path.join('..', 'data_preprocessed')
if not os.path.exists(dir_results):
    os.makedirs(dir_results)

##########
# log file
##########
file_results = os.path.join(dir_results, 'results.txt')
if os.path.isfile(file_results):
    os.remove(file_results)

##########################
# create object for Corpus
##########################
corpus = Corpus()

open(file_results, 'a').write('parse large PAN dataset...' + '\n')
corpus.parse_raw_data(dir_pairs_PAN, dir_label_PAN)

open(file_results, 'a').write('split data set...' + '\n')
corpus.split_data()

##############################
# store results (binary files)
##############################
with open(os.path.join(dir_results, 'dict_author_fandom_doc_train'), 'wb') as f:
    pickle.dump(corpus.dict_author_fandom_doc_train, f)
with open(os.path.join(dir_results, 'dict_author_fandom_doc_cal'), 'wb') as f:
    pickle.dump(corpus.dict_author_fandom_doc_cal, f)
with open(os.path.join(dir_results, 'dict_author_fandom_doc_val'), 'wb') as f:
    pickle.dump(corpus.dict_author_fandom_doc_val, f)

############
# statistics
############

open(file_results, 'a').write('# unique docs: ' + str(len(corpus.unique_docs)) + '\n')
open(file_results, 'a').write('# unique authors: ' + str(len(corpus.authors)) + '\n')
open(file_results, 'a').write('# unique fandoms: ' + str(len(corpus.fandoms)) + '\n')

open(file_results, 'a').write('# docs (train): ' + str(corpus.n_train) + '\n')
open(file_results, 'a').write('# docs (cal): ' + str(corpus.n_cal) + '\n')
open(file_results, 'a').write('# docs (val): ' + str(corpus.n_val) + '\n')
open(file_results, 'a').write('# docs (dropped): ' + str(corpus.n_dropped) + '\n')

open(file_results, 'a').write('# authors (train): ' + str(len(corpus.authors_train)) + '\n')
open(file_results, 'a').write('# authors (cal): ' + str(len(corpus.authors_cal)) + '\n')
open(file_results, 'a').write('# authors (val): ' + str(len(corpus.authors_val)) + '\n')

open(file_results, 'a').write('# fandoms (train): ' + str(len(corpus.fandoms_train)) + '\n')
open(file_results, 'a').write('# fandoms (cal): ' + str(len(corpus.fandoms_cal)) + '\n')
open(file_results, 'a').write('# fandoms (val): ' + str(len(corpus.fandoms_val)) + '\n')

open(file_results, 'a').write('intersection authors (train + cal): ' + str(len(corpus.authors_train.intersection(corpus.authors_cal))) + '\n')
open(file_results, 'a').write('intersection authors (train + val): ' + str(len(corpus.authors_train.intersection(corpus.authors_val))) + '\n')
open(file_results, 'a').write('intersection authors (cal + val): ' + str(len(corpus.authors_cal.intersection(corpus.authors_val))) + '\n')

open(file_results, 'a').write('intersection fandoms (train + cal): ' + str(len(corpus.fandoms_train.intersection(corpus.fandoms_cal))) + '\n')
open(file_results, 'a').write('intersection fandoms (train + val): ' + str(len(corpus.fandoms_train.intersection(corpus.fandoms_val))) + '\n')
open(file_results, 'a').write('intersection fandoms (cal + val): ' + str(len(corpus.fandoms_cal.intersection(corpus.fandoms_val))) + '\n')

open(file_results, 'a').write('finished!' + '\n')
