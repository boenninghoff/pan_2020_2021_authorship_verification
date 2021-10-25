# -*- coding: utf-8 -*-
from sklearn.utils import shuffle
import numpy as np
import pickle
import os
import random


######################################
# class for sampling same-author pairs
######################################
class SameAuthor(object):

    def __init__(self, id):

        # categories with >=2 documents
        self.list_cw2d = []
        # dictionary = {category: documents}
        self.dict_c2d = {}
        # total number of documents
        self.n_docs = 0
        # author id
        self.id = id

    # add document to author
    def add_doc(self, d, c):

        # increase doc counter
        self.n_docs += 1

        if c in self.dict_c2d:
            # add document to existing category
            self.dict_c2d[c].append(d)
            if c not in self.list_cw2d:
                # add category to cw2d-list
                self.list_cw2d.append(c)
        else:
            # new category
            self.dict_c2d[c] = [d]

    # same category
    def sample_SC(self):

        if len(self.list_cw2d) >= 1:
            # shuffle categories, choose one category
            self.list_cw2d = shuffle(self.list_cw2d)
            c = self.list_cw2d[0]
            # shuffle documents in chosen category
            self.dict_c2d[c] = shuffle(self.dict_c2d[c])
            # take two documents, remove them from list
            d_1 = self.dict_c2d[c].pop()
            d_2 = self.dict_c2d[c].pop()
            # decrease doc counter
            self.n_docs -= 2
            # clean-up
            if len(self.dict_c2d[c]) <= 1:
                # remove from cw2d-list if category contains less than 2 documents
                self.list_cw2d.remove(c)
                # delete category if no document remains
                if len(self.dict_c2d[c]) == 0:
                    del self.dict_c2d[c]
            return d_1, d_2
        else:
            return None, None

    # different category
    def sample_DC(self):
        if len(self.dict_c2d) >= 2:

            # shuffle list of categories
            list_c = shuffle(list(self.dict_c2d.keys()))
            # take two categories, remove them from list
            c_1 = list_c.pop()
            c_2 = list_c.pop()
            # shuffle documents
            self.dict_c2d[c_1] = shuffle(self.dict_c2d[c_1])
            self.dict_c2d[c_2] = shuffle(self.dict_c2d[c_2])
            # take two documents
            d_1 = self.dict_c2d[c_1][0]
            d_2 = self.dict_c2d[c_2][0]
            # decrease doc counter
            self.n_docs -= 2
            # clean-up
            del self.dict_c2d[c_1][0], self.dict_c2d[c_2][0]
            if len(self.dict_c2d[c_1]) <= 1:
                if c_1 in self.list_cw2d:
                    self.list_cw2d.remove(c_1)
                # delete category if no document remains
                if len(self.dict_c2d[c_1]) == 0:
                    del self.dict_c2d[c_1]
            if len(self.dict_c2d[c_2]) <= 1:
                if c_2 in self.list_cw2d:
                    self.list_cw2d.remove(c_2)
                # delete category if no document remains
                if len(self.dict_c2d[c_2]) == 0:
                    del self.dict_c2d[c_2]
            return d_1, d_2
        else:
            return None, None

    # get sample for different author object
    def get_sample_for_DA(self):

        if self.n_docs >= 1:
            # get category and document
            c = shuffle(list(self.dict_c2d.keys()))[0]
            self.dict_c2d[c] = shuffle(self.dict_c2d[c])
            d = self.dict_c2d[c][0]
            # decrease doc counter
            self.n_docs -= 1
            # clean-up
            del self.dict_c2d[c][0]
            if len(self.dict_c2d[c]) <= 1:
                # remove from cw2d-list if category contains less than 2 documents
                if c in self.list_cw2d:
                    self.list_cw2d.remove(c)
                # delete category if no document remains
                if len(self.dict_c2d[c]) == 0:
                    del self.dict_c2d[c]
            return c, d, self.id
        else:
            return None, None, None


###################################
# class for different-authors pairs
###################################
class DifferentAuthor(object):

    def __init__(self):

        # categories with >=2 documents
        self.list_cw2d = []
        # dictionary = {category: documents}
        self.dict_c2d = {}
        # total number of documents
        self.n_docs = 0

    # add document to different author object
    def add_doc(self, d, c, id):

        # increase doc counter
        self.n_docs += 1

        if c in self.dict_c2d:
            # add document to existing category
            self.dict_c2d[c].append((d, id))
            if c not in self.list_cw2d:
                # add document to cw2d-list
                self.list_cw2d.append(c)
        else:
            # new category
            self.dict_c2d[c] = [(d, id)]

    # sample same category
    def sample_SC(self):

        if len(self.list_cw2d) >= 1:

            # shuffle categories, choose one category
            self.list_cw2d = shuffle(self.list_cw2d)
            c = self.list_cw2d[0]
            # shuffle documents in chosen category
            self.dict_c2d[c] = shuffle(self.dict_c2d[c])
            # take two documents, remove them from list
            d_1, id_1 = self.dict_c2d[c].pop()
            d_2, id_2 = self.dict_c2d[c].pop()
            # decrease doc counter
            self.n_docs -= 2
            # clean-up
            if len(self.dict_c2d[c]) <= 1:
                # remove from cw2d-list if category contains less than 2 documents
                self.list_cw2d.remove(c)
                # delete category if no document remains
                if len(self.dict_c2d[c]) == 0:
                    del self.dict_c2d[c]

            return d_1, d_2, id_1, id_2
        else:
            return None, None, None, None

    # sample different categories
    def sample_DC(self):

        if len(self.dict_c2d) >= 2:

            # shuffle list of categories
            list_c = shuffle(list(self.dict_c2d.keys()))
            # take two categories, remove them from list
            c_1 = list_c.pop()
            c_2 = list_c.pop()
            # shuffle documents
            self.dict_c2d[c_1] = shuffle(self.dict_c2d[c_1])
            self.dict_c2d[c_2] = shuffle(self.dict_c2d[c_2])
            # take two documents
            d_1, id_1 = self.dict_c2d[c_1][0]
            d_2, id_2 = self.dict_c2d[c_2][0]
            # decrease doc counter
            self.n_docs -= 2
            # clean-up
            del self.dict_c2d[c_1][0], self.dict_c2d[c_2][0]
            if len(self.dict_c2d[c_1]) <= 1:
                if c_1 in self.list_cw2d:
                    self.list_cw2d.remove(c_1)
                # delete category if no document remains
                if len(self.dict_c2d[c_1]) == 0:
                    del self.dict_c2d[c_1]
            if len(self.dict_c2d[c_2]) <= 1:
                if c_2 in self.list_cw2d:
                    self.list_cw2d.remove(c_2)
                # delete category if no document remains
                if len(self.dict_c2d[c_2]) == 0:
                    del self.dict_c2d[c_2]
            return d_1, d_2, id_1, id_2
        else:
            return None, None, None, None


#################################
# make pairs for single partition
#################################
def sample_pairs_single_epoch(author_domain_doc, 
                              delta_1=0.5, delta_2=0.5, delta_3=0.5, 
                              only_SADF_and_DASF=False,
                              make_balanced=False,
                              balance_factor=1.0,
                              ):
    
    """
    Input:
        author_domain_doc:      dictionary with sorted documents 
        delta_1:                threshold (SA vs. DA) 
        delta_2:                threshold (SA_SF vs. SA_DF)
        delta_3:                threshold (DA_SF vs. DA_DF)
        only_SADF_and_DASF:     if true, return only SA_DF and DA_SF pairs
        make_balanced:          if true, return (nearly) balanced dataset
        balance_factor:         balancing factor, if 1.0, all subsets contain the same number of trials
        
    Output:
        docs_L, docs_R:         document pairs
        labels_a:               authorship label (0 or 1)
        labels_c:               category/fandom label (0 or 1)
            
    """
    
    # author list
    author_list = list(author_domain_doc.keys())

    # define list for all subsets
    SA_SF = []
    SA_DF = []
    DA_SF = []
    DA_DF = []
    
    # dictionary for same-author objects
    author_models = {}
    # create different-authors object
    author_diff = DifferentAuthor()
    
    # create same-author objects
    for a in author_list:
        # create same-author object for current author
        author_models[a] = SameAuthor(a)
        # add docs
        for c in author_domain_doc[a].keys():
            for d in author_domain_doc[a][c]:
                author_models[a].add_doc(d, c)
    
    # start re-sampling
    while bool(author_models):

        author_list = shuffle(list(author_models.keys()))

        for a in author_list:

            r1 = random.uniform(0, 1)

            if r1 < delta_1:

                r2 = random.uniform(0, 1)

                ###################
                # sample SA_DF pair
                ###################
                if r2 < delta_2:
                    d1, d2 = author_models[a].sample_DC()
                    if bool(d1):
                        SA_DF.append((d1, d2, 1, 0))

                ###################
                # sample SA_SF pair
                ###################
                else:
                    d1, d2 = author_models[a].sample_SC()
                    if bool(d1):
                        SA_SF.append((d1, d2, 1, 1))

            else:
                ###################
                # sample doc for DA
                ###################
                c, d, id = author_models[a].get_sample_for_DA()
                if bool(c):
                    author_diff.add_doc(d, c, id)

            # clean
            if author_models[a].n_docs == 0:
                del author_models[a]

    while author_diff.n_docs > 1:

        r3 = random.uniform(0, 1)

        ###################
        # sample DA_SF pair
        ###################
        if r3 < delta_3:
            d1, d2, id_1, id_2 = author_diff.sample_SC()
            if bool(d1):
                if id_1 == id_2:
                    SA_SF.append((d1, d2, 1, 1))
                else:
                    DA_SF.append((d1, d2, 0, 1))

        ###################
        # sample DA_DF pair
        ###################
        else:
            d1, d2, id_1, id_2 = author_diff.sample_DC()
            if bool(d1):
                if id_1 == id_2:
                    SA_DF.append((d1, d2, 1, 0))
                else:
                    DA_DF.append((d1, d2, 0, 0))

    #####################
    # prepare final pairs
    #####################
    docs_L = []
    docs_R = []
    labels_a = []
    labels_c = []
    
    # shuffle datasets
    SA_SF = shuffle(SA_SF)
    SA_DF = shuffle(SA_DF)
    DA_SF = shuffle(DA_SF)
    DA_DF = shuffle(DA_DF)
    
    # make balanced datasets (for evaluation)   
    if make_balanced:
        if only_SADF_and_DASF:
            n = min(len(SA_DF), len(DA_SF))
            n = int(balance_factor * n)
            SA_DF = SA_DF[:n]
            DA_SF = DA_SF[:n]
        else:
            n = min(len(SA_SF), len(SA_DF), len(DA_SF), len(DA_DF))
            n = int(balance_factor * n)
            SA_SF = SA_SF[:n]
            SA_DF = SA_DF[:n]
            DA_SF = DA_SF[:n]
            DA_DF = DA_DF[:n]

    # consider only SA_DF and DA_SF pairs (for evaluation)
    if only_SADF_and_DASF:    
        pairs = SA_DF + DA_SF
    else:
        pairs = SA_SF + SA_DF + DA_SF + DA_DF
    # shuffle
    pairs = shuffle(pairs)
    pairs = shuffle(pairs)

    for pair in pairs:

        doc_1, doc_2, l_a, l_c = pair

        ########
        # labels
        ########
        labels_a.append(l_a)
        labels_c.append(l_c)

        ######
        # docs
        ######
        r = random.uniform(0, 1)
        if r < 0.5:
            docs_L.append(doc_1)
            docs_R.append(doc_2)
        else:
            docs_L.append(doc_2)
            docs_R.append(doc_1)

    return docs_L, docs_R, labels_a, labels_c




