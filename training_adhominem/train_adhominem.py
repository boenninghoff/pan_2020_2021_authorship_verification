# -*- coding: utf-8 -*-
from adhominem import AdHominem
import pickle
import os
import argparse


def main():

    parser = argparse.ArgumentParser(description='AdHominem for PAN 2020 and 2021')
    parser.add_argument('-D_c', default=10, type=int)  # character embedding dimension
    parser.add_argument('-D_r', default=30, type=int)  # character representation dimension
    parser.add_argument('-D_w', default=300, type=int)  # dimension of word embeddings
    parser.add_argument('-D_s', default=50, type=int)  # dimension of sentence embeddings
    parser.add_argument('-D_d', default=50, type=int)  # dimension of document embedding
    parser.add_argument('-D_lev', default=60, type=int)  # LEV dimension
    parser.add_argument('-D_plda', default=40, type=int)  # dimension reduction for BFS
    parser.add_argument('-w', default=4, type=int)  # length of 1D-CNN sliding window
    #
    parser.add_argument('-T_c', default=15, type=int)  # maximum number of characters per words
    parser.add_argument('-T_w', default=30, type=int)  # maximum number of words per sentence
    parser.add_argument('-T_s', default=210, type=int)  # maximum number of sentences per document
    #
    parser.add_argument('-t_s', default=0.91, type=float)  # boundary for similar pairs
    parser.add_argument('-t_d', default=0.09, type=float)  # boundary for dissimilar pairs
    #
    parser.add_argument('-lr_start', default=0.0006, type=float)  # initial learning rate
    parser.add_argument('-lr_end', default=0.0002, type=float)  # lower bound for learning rate
    parser.add_argument('-lr_epoch', default=30, type=float)  # epoch, when approaching the lower bound
    #
    parser.add_argument('-epochs', default=50, type=int)  # total number of epochs
    parser.add_argument('-batch_size', default=40, type=int)  # batch size for training
    parser.add_argument('-batch_size_dev', default=40, type=int)  # batch size for evaluation
    #
    parser.add_argument('-keep_prob_cnn', default=0.8, type=float)  # dropout for 1D-CNN
    parser.add_argument('-keep_prob_lstm', default=0.9, type=float)  # variational dropout for BiLSTM layer
    parser.add_argument('-keep_prob_att', default=0.9, type=float)  # dropout for attention layer
    parser.add_argument('-keep_prob_metric', default=0.8, type=float)  # dropout for final DML layer
    parser.add_argument('-keep_prob_bfs', default=0.8, type=float)  # dropout for BFS layer
    parser.add_argument('-keep_prob_ual', default=0.8, type=float)  # dropout for UAL layer
    #
    parser.add_argument('-train_kernel', default=True, type=bool)  # train kernel parameters for DML
    parser.add_argument('-reg_ual', default=0.125, type=float)  # regularization hyper-parameter for UAL
    parser.add_argument('-stop_gradient_bfs', default=True, type=bool)  # backprop of bfs loss
    parser.add_argument('-stop_gradient_ual', default=True, type=bool)  # backprop of ual loss
    parser.add_argument('-gsl_bfs', default=0.0, type=float)  # gradient scaling (only if stop_gradient_bfs=False)
    parser.add_argument('-gsl_ual', default=0.0, type=float)  # gradient scaling (only if stop_gradient_ual=False)
    #
    parser.add_argument('-hop_length', default=26, type=int)  # hop length for sliding windowing
    parser.add_argument('-num_wrd_embeddings', default=5000, type=int)  # vocabulary size for tokens
    parser.add_argument('-num_chr_embeddings', default=300, type=int)  # vocabulary size for characters
    #
    hyper_parameters = vars(parser.parse_args())

    # create folder for results
    dir_results = os.path.join('..', 'results_adhominem')
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)

    # load dev set
    with open(os.path.join('..', 'data_preprocessed', "pairs_dev"), 'rb') as f:
        docs_L, docs_R, labels, _ = pickle.load(f)
    dev_set = (docs_L, docs_R, labels)
    hyper_parameters['N_dev'] = len(labels)

    # load vocabularies and initialized word embeddings
    path_wrd = os.path.join('..', 'data_preprocessed',
                            'vocab_wrd_' + str(hyper_parameters['num_wrd_embeddings'])
                                               + "_" + str(hyper_parameters['D_w']))
    with open(path_wrd, 'rb') as f:
        V_w, E_w = pickle.load(f)
    path_chr = os.path.join('..', 'data_preprocessed', 'vocab_chr_' + str(hyper_parameters['num_chr_embeddings']))
    with open(path_chr, 'rb') as f:
        V_c = pickle.load(f)

    # add vocabularies to dictionary
    hyper_parameters['V_w'] = V_w
    hyper_parameters['V_c'] = V_c

    # file to store results epoch-wise
    file_results = os.path.join(dir_results, 'results.txt')
    # temporary file to store results batch-wise
    file_tmp = os.path.join(dir_results, 'tmp.txt')

    # delete already existing files
    if os.path.isfile(file_results):
        os.remove(file_results)
    if os.path.isfile(file_tmp):
        os.remove(file_tmp)

    # write hyper-parameters setup into file (results.txt)
    open(file_results, 'a').write('\n'
                                  + '--------------------------------------------------------------------------------'
                                  + '\nPARAMETER SETUP:\n'
                                  + '--------------------------------------------------------------------------------'
                                  + '\n'
                                  )
    for hp in sorted(hyper_parameters.keys()):
        if hp in ['V_c', 'V_w']:
            open(file_results, 'a').write('num ' + hp + ': ' + str(len(hyper_parameters[hp])) + '\n')
        else:
            open(file_results, 'a').write(hp + ': ' + str(hyper_parameters[hp]) + '\n')

    # load neural network model
    adhominem = AdHominem(hyper_parameters=hyper_parameters,
                          E_w_init=E_w,
                          )
    # start training
    adhominem.train_model(dev_set, file_results, file_tmp, dir_results)
    # close session
    adhominem.sess.close()


if __name__ == '__main__':
    main()
