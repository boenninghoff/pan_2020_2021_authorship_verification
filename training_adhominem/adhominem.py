import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
import numpy as np
import pickle
import random
import os
from math import ceil
import datetime
from sklearn.utils import shuffle
import sys
sys.path.append(os.path.join('..', 'helper_functions'))
from evaluate import evaluate_all
from reliability_diagrams import compute_calibration
from resample_pairs_train import resample_pairs_train


class AdHominem():
    """
        AdHominem describes a Siamese network topology for (binary) authorship verification, also known as pairwise
        (1:1) forensic text comparison [1, 2]. This implementation was developed for the The PAN 2020/2021 authorship
        verification shared tasks. It represents a hierarchical fusion of three well-known approaches into a single
        end-to-end learning procedure: A deep metric learning framework at the bottom aims to learn a pseudo-metric
        that maps a document of variable length onto a fixed-sized 'linguistic embedding vetor' (LEV). Next, we
        incorporate a probabilistic layer to perform Bayes factor scoring (BFS) layer in the learned metric space [3].
        Finally, an uncertainty adaptation layer (UAL) is used to find and correct wrongly classified trials of the
        BFS layer, to model its noise behavior, and to return re-calibrated posteriors [4, 5].
        As with most deep-learning approaches, the success of the proposed architecture depends heavily on the
        availability of a large collection of text samples with many examples of representative variations in writing
        style. The size of the train set has been increased synthetically by dissembling all predefined document pairs
        and re-sampling new same-author and different-author pairs in each epoch [2, 3, 4].

        References:
        [1] Benedikt Boenninghoff, Robert M. Nickel, Steffen Zeiler, Dorothea Kolossa, 'Similarity Learning for
            Authorship Verification in Social Media', IEEE ICASSP 2019.
        [2] Benedikt Boenninghoff, Steffen Hessler, Dorothea Kolossa, Robert M. Nickel 'Explainable Authorship
            Verification in Social Media via Attention-based Similarity Learning', IEEE BigData 2019.
        [3] Benedikt Boenninghoff, Julian Rupp, Dorothea Kolossa, Robert M. Nickel 'Deep Bayes Factor Scoring For
            Authorship Verification', PAN Workshop Notebook at CLEF 2020.
        [4] Benedikt Boenninghoff, Dorothea Kolossa, Robert M. Nickel 'Self-Calibrating Neural-Probabilistic Model for
            Authorship Verification Under Covariate Shift', CLEF 2021.
        [5] Benedikt Boenninghoff, Robert M. Nickel, Dorothea Kolossa, 'O2D2: Out-Of-Distribution Detector to Capture
            Undecidable Trials in Authorship Verification', PAN Workshop Notebook at CLEF 2021.

    """

    def __init__(self, hyper_parameters, E_w_init):

        # reset graph
        tf.reset_default_graph()

        # hyper-parameters
        self.hyper_parameters = hyper_parameters

        # placeholders for input variables
        self.placeholders, self.thetas_E = self.initialize_placeholders(E_w_init)

        # batch size
        self.B = tf.shape(self.placeholders['e_w_L'])[0]

        # trainable parameters
        self.theta = self.initialize_parameters()
        # initialize dropout
        self.dropout = self.initialize_dropout()

        ##########################################
        # document embeddings (feature extraction)
        ##########################################
        with tf.variable_scope('feature_extraction_doc2vec'):
            e_c = tf.concat([self.placeholders['e_c_L'], self.placeholders['e_c_R']], axis=0)
            e_w = tf.concat([self.placeholders['e_w_L'], self.placeholders['e_w_R']], axis=0)
            N_w = tf.concat([self.placeholders['N_w_L'], self.placeholders['N_w_R']], axis=0)
            N_s = tf.concat([self.placeholders['N_s_L'], self.placeholders['N_s_R']], axis=0)
            # doc2vec
            e_d = self.feature_extraction(e_c, e_w, N_w, N_s)

        ############################
        # deep metric learning (DML)
        ############################
        with tf.variable_scope('deep_metric_learning_a'):
            self.lev_a_L, self.lev_a_R = self.metric_layer(e_d,
                                                           self.dropout['metric_a'],
                                                           self.theta['metric_a'],
                                                           stop_gradient=False,
                                                           )
        with tf.variable_scope('deep_metric_learning_f'):
            self.lev_f_L, self.lev_f_R = self.metric_layer(e_d,
                                                           self.dropout['metric_f'],
                                                           self.theta['metric_f'],
                                                           stop_gradient=True,
                                                           )

        ###################################################
        # kernel distance for probabilstic contrastive loss
        ###################################################
        with tf.variable_scope('Euclidean_distance_and_kernel_function_a'):
            self.pred_dml_a = self.compute_kernel_distance(self.lev_a_L, self.lev_a_R,
                                                           alpha=self.theta["loss_dml_a"]["alpha"],
                                                           beta=self.theta["loss_dml_a"]["beta"],
                                                           )
        with tf.variable_scope('Euclidean_distance_and_kernel_function_f'):
            self.pred_dml_f = self.compute_kernel_distance(self.lev_f_L, self.lev_f_R,
                                                           alpha=0.09,
                                                           beta=3.0 / 2.0,
                                                           )

        ##################################
        # Bayes factor scoring (BFS) layer
        ##################################
        with tf.variable_scope('Bayes_factor_scoring_a'):
            self.scores_bfs_a, self.pred_bfs_a, \
            self.H_W_a, self.H_B_a = self.bfs_layer(self.lev_a_L,
                                                    self.lev_a_R,
                                                    self.theta['bfs_a_1'],
                                                    self.theta['bfs_a_2'],
                                                    self.dropout['bfs_a'],
                                                    stop_gradient=self.hyper_parameters["stop_gradient_bfs"],
                                                    )
        with tf.variable_scope('Bayes_factor_scoring_f'):
            self.scores_bfs_f, self.pred_bfs_f, \
            self.H_W_f, self.H_B_f = self.bfs_layer(self.lev_f_L,
                                                    self.lev_f_R,
                                                    self.theta['bfs_f_1'],
                                                    self.theta['bfs_f_2'],
                                                    self.dropout['bfs_f'],
                                                    stop_gradient=self.hyper_parameters["stop_gradient_bfs"],
                                                    )

        ####################################
        # uncertainty adaptation layer (UAL)
        ####################################
        self.loss_ual_a, self.reg_ual_a, self.pred_ual_a, self.conf_matrix_a \
            = self.ual_layer(self.theta['ual_a'],
                             self.dropout['ual_a'],
                             self.pred_bfs_a,
                             self.lev_a_L,
                             self.lev_a_R,
                             stop_gradient=self.hyper_parameters["stop_gradient_ual"],
                             )

        ##############################
        # loss functions and optimizer
        ##############################

        # loss functions for authorship
        self.loss_metric_a = self.loss_function_dml(pred=self.pred_dml_a, labels=self.placeholders['labels_a'])
        self.loss_plda_a = self.loss_function_bfs(scores=self.scores_bfs_a, labels=self.placeholders['labels_a'])
        self.loss_ual_a = self.loss_ual_a + self.hyper_parameters['reg_ual'] * self.reg_ual_a
        self.loss_a = self.loss_metric_a + self.loss_plda_a + self.loss_ual_a

        # loss functions for fandoms
        self.loss_metric_f = self.loss_function_dml(pred=self.pred_dml_f, labels=self.placeholders['labels_f'])
        self.loss_plda_f = self.loss_function_bfs(scores=self.scores_bfs_f, labels=self.placeholders['labels_f'])
        self.loss_f = self.loss_metric_f + self.loss_plda_f

        # joint loss
        self.loss = self.loss_a + self.loss_f

        # optimizer
        self.optimizer, self.step = self.prepare_optimizer()

        ################
        # launch session
        ################
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    ########################
    # gradient scaling layer
    ########################
    @staticmethod
    def scale_gradient(x, gsl_beta=1.0):
        forward_path = tf.stop_gradient(x * tf.cast(1.0 - gsl_beta, tf.float32))
        backward_path = x * tf.cast(gsl_beta, tf.float32)
        return forward_path + backward_path

    ####################################
    # uncertainty adaptation layer (UAL)
    ####################################
    def ual_layer(self, theta, dropout, pred_bfs, lev_L, lev_R, stop_gradient=True):

        if stop_gradient:
            lev_L = tf.stop_gradient(lev_L)
            lev_R = tf.stop_gradient(lev_R)
            pred_bfs = tf.stop_gradient(pred_bfs)
        else:
            lev_L = self.scale_gradient(lev_L, gsl_beta=self.hyper_parameters['gsl_ual'])
            lev_R = self.scale_gradient(lev_R, gsl_beta=self.hyper_parameters['gsl_ual'])
            pred_bfs = self.scale_gradient(pred_bfs, gsl_beta=self.hyper_parameters['gsl_ual'])

        labels = tf.reshape(tf.cast(self.placeholders['labels_a'], dtype=tf.int32), shape=[self.B])

        y = tf.square(lev_L - lev_R)

        # apply dropout
        is_training = self.placeholders['is_training']
        y = tf.cond(tf.equal(is_training, tf.constant(True)),
                    lambda: tf.multiply(dropout['ual_1'], y),
                    lambda: y,
                    )

        # dense layer
        y = tf.nn.xw_plus_b(y, theta["W1"], theta["b1"])
        y = tf.nn.tanh(y)

        # apply dropout
        y = tf.cond(tf.equal(is_training, tf.constant(True)),
                    lambda: tf.multiply(dropout['ual_2'], y),
                    lambda: y,
                    )

        # compute confusion matrix, shape = [B, 2 * 2]
        logits = tf.nn.xw_plus_b(y, theta["W2"], theta["b2"])
        # shape = [B, 2, 2]
        logits = tf.reshape(logits, shape=[-1, 2, 2])

        conf_matrix = tf.nn.softmax(logits, axis=1)
        pred_bfs = tf.concat([tf.subtract(1.0, pred_bfs), pred_bfs], axis=1)

        # uncertainty adaptation, shape = [B, 2, 2] --> [B, 2]
        pred_ual = tf.squeeze(tf.matmul(conf_matrix, tf.expand_dims(pred_bfs, axis=2)), axis=2)

        # maximum entropy regularization
        p = tf.maximum(tf.reshape(conf_matrix, shape=[self.B, 4]), 1e-6)
        reg = tf.reduce_sum(tf.multiply(tf.log(p), p), axis=1)
        reg = tf.reduce_mean(reg)

        # loss
        idx = tf.stack([tf.range(0, self.B, 1), labels], axis=1)
        loss_ual = tf.gather_nd(pred_ual, idx)
        loss_ual = tf.reduce_mean(-tf.log(loss_ual + 1e-8))

        return loss_ual, reg, pred_ual, conf_matrix

    ###################
    # prepare optimizer
    ###################
    def prepare_optimizer(self):

        # global step counter
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        # optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.placeholders['lr'])

        # local gradient normalization
        grads_and_vars = opt.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_norm(grad, 5.0), var) for grad, var in grads_and_vars]
        optimizer = opt.apply_gradients(clipped_grads_and_vars, global_step=global_step)

        return optimizer, global_step

    ########################################
    # final deep metric learning (DML) layer
    ########################################
    def metric_layer(self, e_d, dropout, theta, stop_gradient=False):

        if stop_gradient:
            e_d = tf.stop_gradient(e_d)

        is_training = self.placeholders['is_training']
        dropout_mask = tf.concat([dropout['metric_1'],
                                  dropout['metric_1']],
                                 axis=0)

        # apply dropout
        y = tf.cond(tf.equal(is_training, tf.constant(True)),
                    lambda: tf.multiply(dropout_mask, e_d),
                    lambda: e_d,
                    )
        # fully-connected layer
        y = tf.nn.xw_plus_b(y,
                            theta['W'],
                            theta['b'],
                            )
        # nonlinear output
        y = tf.nn.tanh(y)

        return y[:self.B, :], y[self.B:, :]

    ##################################
    # Bayes factor scoring (BFS) layer
    ##################################
    def bfs_layer(self, lev_L, lev_R, theta_bfs_1, theta_bfs_2, dropout, stop_gradient=True):

        def makeS(C, D):
            diagC = tf.math.exp(tf.diag_part(C))
            maskC = tf.ones_like(C) - tf.eye(D)
            maskC = tf.matrix_band_part(maskC, 0, -1)
            C = tf.multiply(maskC, C) + tf.diag(diagC)
            S = tf.matmul(C, C, transpose_b=True) + 1e-6 * tf.eye(D)
            return S

        def bilinear_form(x1, X, x2):
            # [None, D] x [D, D] = [None, D]
            x1X = tf.matmul(x1, X)
            # [None, 1, D] x [None, D, 1] = [None, 1, 1]
            x1X = tf.expand_dims(x1X, axis=2)
            x1Xx2 = tf.matmul(tf.expand_dims(x2, axis=1), x1X)
            # [None, 1]
            x1Xx2 = tf.squeeze(x1Xx2, axis=2)
            return x1Xx2

        if stop_gradient:
            lev_L = tf.stop_gradient(lev_L)
            lev_R = tf.stop_gradient(lev_R)
        else:
            lev_L = self.scale_gradient(lev_L, gsl_beta=self.hyper_parameters['gsl_bfs'])
            lev_R = self.scale_gradient(lev_R, gsl_beta=self.hyper_parameters['gsl_bfs'])

        # apply dropout
        is_training = self.placeholders['is_training']
        x1 = tf.cond(tf.equal(is_training, tf.constant(True)),
                     lambda: tf.multiply(dropout['bfs_1'], lev_L),
                     lambda: lev_L,
                     )
        x2 = tf.cond(tf.equal(is_training, tf.constant(True)),
                     lambda: tf.multiply(dropout['bfs_1'], lev_R),
                     lambda: lev_R,
                     )

        # fully-connected layer
        x1 = tf.nn.xw_plus_b(x1,
                             theta_bfs_1['W'],
                             theta_bfs_1['b'],
                             )
        x2 = tf.nn.xw_plus_b(x2,
                             theta_bfs_1['W'],
                             theta_bfs_1['b'],
                             )
        x1 = tf.nn.relu(x1)
        x2 = tf.nn.relu(x2)

        # PLDA layer
        B = theta_bfs_2['B']
        B = makeS(B, self.hyper_parameters['D_plda'])
        W = theta_bfs_2['W']
        W = makeS(W, self.hyper_parameters['D_plda'])
        mu = theta_bfs_2['mu']

        A_tilde = tf.matrix_inverse(B + 2 * W + 1e-6 * tf.eye(self.hyper_parameters['D_plda']))
        G_tilde = tf.matrix_inverse(B + W + 1e-6 * tf.eye(self.hyper_parameters['D_plda']))

        B_mu = tf.matmul(B, mu)
        mu_B_mu = tf.matmul(mu, B_mu, transpose_a=True)
        W_AmG = tf.matmul(W, A_tilde - G_tilde, transpose_a=True)
        mu_B_Am2G = tf.matmul(tf.matmul(B_mu, A_tilde - 2 * G_tilde, transpose_a=True), B_mu)

        A = 0.5 * tf.matmul(tf.matmul(W, A_tilde, transpose_a=True), W)
        G = 0.5 * tf.matmul(W_AmG, W)
        c = tf.matmul(W_AmG, B_mu)

        k_tilde = 2 * tf.linalg.logdet(G_tilde) - tf.linalg.logdet(B) - tf.linalg.logdet(A_tilde) + mu_B_mu
        k = k_tilde + 0.5 * mu_B_Am2G

        x1Ax2 = bilinear_form(x1, A, x2)
        x2Ax1 = bilinear_form(x2, A, x1)
        x1Gx1 = bilinear_form(x1, G, x1)
        x2Gx2 = bilinear_form(x2, G, x2)
        x1x2c = tf.matmul(tf.add(x1, x2), c)

        score = x1Ax2 + x2Ax1 + x1Gx1 + x2Gx2 + x1x2c + k

        pred = tf.nn.sigmoid(score)

        # compute entropy
        H_W = tf.linalg.logdet(tf.matrix_inverse(W))
        H_B = tf.linalg.logdet(tf.matrix_inverse(B))

        return score, pred, H_W, H_B

    ################################################
    # initialize all placeholders and look-up tables
    ################################################
    def initialize_placeholders(self, E_w_init):

        T_c = self.hyper_parameters['T_c']
        T_w = self.hyper_parameters['T_w']
        T_s = self.hyper_parameters['T_s']
        D_c = self.hyper_parameters['D_c']
        D_w = self.hyper_parameters['D_w']
        V_c = self.hyper_parameters['V_c']

        r = 0.03

        # input character placeholder
        x_c_L = tf.placeholder(dtype=tf.int32,
                               shape=[None, T_s, T_w, T_c],
                               name='x_c_L',
                               )
        x_c_R = tf.placeholder(dtype=tf.int32,
                               shape=[None, T_s, T_w, T_c],
                               name='x_c_R',
                               )

        # initialize embedding matrix for characters
        with tf.variable_scope('character_embedding_matrix'):
            # zero-padding embedding
            E_c_0 = tf.zeros(shape=[1, D_c], dtype=tf.float32)
            # trainable embeddings
            E_c_1 = tf.get_variable(name='E_c_1',
                                    shape=[len(V_c) - 1, D_c],
                                    initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                    trainable=True,
                                    dtype=tf.float32,
                                    )
            # concatenate special-token embeddings + regular-token embeddings
            E_c = tf.concat([E_c_0, E_c_1], axis=0)

        # character embeddings, shape=[B, T_s, T_w, T_c, D_c]
        e_c_L = tf.nn.embedding_lookup(E_c, x_c_L)
        e_c_R = tf.nn.embedding_lookup(E_c, x_c_R)

        # word-based placeholder for two documents
        x_w_L = tf.placeholder(dtype=tf.int32, shape=[None, T_s, T_w], name='x_w_L')
        x_w_R = tf.placeholder(dtype=tf.int32, shape=[None, T_s, T_w], name='x_w_R')

        # true sentence / document lengths
        N_w_L = tf.placeholder(dtype=tf.int32, shape=[None, T_s], name='N_w_L')
        N_w_R = tf.placeholder(dtype=tf.int32, shape=[None, T_s], name='N_w_R')
        N_s_L = tf.placeholder(dtype=tf.int32, shape=[None], name='N_s_L')
        N_s_R = tf.placeholder(dtype=tf.int32, shape=[None], name='N_s_R')

        # matrix for word embeddings, shape=[len(V_w), D_w]
        with tf.variable_scope('word_embedding_matrix'):
            # zero-padding embedding
            E_w_0 = tf.zeros(shape=[1, D_w], dtype=tf.float32)
            # <UNK> embedding
            E_w_1 = tf.Variable(E_w_init[1, :].reshape((1, D_w)),
                                name='E_w_1',
                                trainable=True,
                                dtype=tf.float32,
                                )
            # pre-trained word embedding
            E_w_2 = tf.Variable(E_w_init[2:, :],
                                name='E_w_2',
                                trainable=True,
                                dtype=tf.float32,
                                )
            # concatenate special-token embeddings + regular-token embeddings
            E_w = tf.concat([E_w_0, E_w_1, E_w_2], axis=0)

        # word embeddings, shape=[B, T_s, T_w, D_w]
        e_w_L = tf.nn.embedding_lookup(E_w, x_w_L)
        e_w_R = tf.nn.embedding_lookup(E_w, x_w_R)

        ####################
        # training variables
        ####################
        # labels
        labels_a = tf.placeholder(tf.float32, shape=[None, 1], name='labels_a')
        labels_f = tf.placeholder(tf.float32, shape=[None, 1], name='labels_f')

        # training mode (for dropout regularization)
        is_training = tf.placeholder(dtype=tf.bool, name='training_mode')
        # learning rate
        lr = tf.placeholder(tf.float32, [], name='lr')

        #############
        # make tuples
        #############
        placeholders = {'x_c_L': x_c_L,
                        'x_c_R': x_c_R,
                        #
                        'e_c_L': e_c_L,
                        'e_c_R': e_c_R,
                        #
                        'x_w_L': x_w_L,
                        'x_w_R': x_w_R,
                        #
                        'e_w_L': e_w_L,
                        'e_w_R': e_w_R,
                        #
                        'N_w_L': N_w_L,
                        'N_w_R': N_w_R,
                        'N_s_L': N_s_L,
                        'N_s_R': N_s_R,
                        #
                        'labels_a': labels_a,
                        'labels_f': labels_f,
                        #
                        'is_training': is_training,
                        'lr': lr,
                        }

        thetas_E = {'E_c_1': E_c_1,
                    'E_w_1': E_w_1,
                    'E_w_2': E_w_2,
                    }

        return placeholders, thetas_E

    ################################################
    # feature extraction: words-to-document encoding
    ################################################
    def feature_extraction(self, e_c, e_w, N_w, N_s):

        with tf.variable_scope('characters_to_word_encoding'):
            r_c = self.cnn_layer_cw(e_c)
        with tf.variable_scope('words_to_sentence_encoding'):
            e_cw = tf.concat([e_w, r_c], axis=3)
            h_w = self.bilstm_layer_ws(e_cw, N_w)
            e_s = self.att_layer_ws(h_w, N_w)
        with tf.variable_scope('sentences_to_document_encoding'):
            h_s = self.bilstm_layer_sd(e_s, N_s)
            e_d = self.att_layer_sd(h_s, N_s)

        return e_d

    ##########################################
    # compute distance between feature vectors
    ##########################################
    @staticmethod
    def compute_kernel_distance(lev_L, lev_R, alpha, beta):
        # define euclidean distance, shape = (B, D_h)
        distance = tf.subtract(lev_L, lev_R)
        distance = tf.square(distance) + 1e-8
        # shape = (B, 1)
        distance = tf.reduce_sum(distance, 1, keepdims=True) + 1e-8
        # kernel
        pred = tf.math.pow(x=distance, y=beta)
        pred = tf.math.exp(-alpha * pred)
        pred = pred + 1e-8
        return pred

    #####################
    # estimate new labels
    #####################
    def compute_labels(self, pred, thr=0.5):
        # numpy array for estimated labels
        labels_hat = np.ones(pred.shape, dtype=np.float32)
        # dissimilar pairs --> 0, similar pairs --> 1
        labels_hat[pred <= thr] = 0
        return labels_hat

    ###############
    # loss function
    ###############
    def loss_function_dml(self, pred, labels):

        t_s = self.hyper_parameters['t_s']
        t_d = self.hyper_parameters['t_d']

        # define contrastive loss:
        l0 = tf.multiply(tf.subtract(1.0, labels), tf.square(tf.maximum(tf.subtract(pred, t_d), 0.0)))
        l1 = tf.multiply(labels, tf.square(tf.maximum(tf.subtract(t_s, pred), 0.0)))
        loss = tf.add(l0, l1)
        loss = tf.reduce_mean(loss)

        return loss

    @staticmethod
    def loss_function_bfs(scores, labels):

        # define binary cross entropy loss
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=scores, labels=labels)
        loss = tf.reduce_mean(loss)

        return loss

    ########################################
    # 1D-CNN for characters-to-word encoding
    ########################################
    def cnn_layer_cw(self, e_c):

        T_s = self.hyper_parameters['T_s']
        T_w = self.hyper_parameters['T_w']
        T_c = self.hyper_parameters['T_c']
        h = self.hyper_parameters['w']
        D_c = self.hyper_parameters['D_c']
        D_r = self.hyper_parameters['D_r']

        is_training = self.placeholders['is_training']
        dropout_mask = tf.concat([self.dropout['cnn'],
                                  self.dropout['cnn']],
                                 axis=0)

        # dropout and zero-padding
        # reshape: [B, T_s, T_w, T_c, D_c] --> [B * T_s * T_w, T_c, D_c]
        e_c = tf.reshape(e_c, shape=[2 * self.B * T_s * T_w, T_c, D_c])
        # dropout
        e_c = tf.cond(tf.equal(is_training, tf.constant(True)),
                      lambda: tf.multiply(dropout_mask, e_c),
                      lambda: e_c,
                      )
        # zero-padding, shape = [B * T_s * T_w, T_c + 2 * (h-1), D_c]
        e_c = tf.pad(e_c,
                     tf.constant([[0, 0], [h - 1, h - 1], [0, 0]]),
                     mode='CONSTANT',
                     )

        # 1D convolution
        # shape = [B * T_s * T_w, T_c + 2 * (h-1) - h + 1, D_r] = [B * T_s * T_w, T_c + h - 1, D_r]
        r_c = tf.nn.conv1d(e_c,
                           self.theta['cnn']['W'],
                           stride=1,
                           padding='VALID',
                           name='chraracter_1D_cnn',
                           )
        # apply bias term
        r_c = tf.nn.bias_add(r_c, self.theta['cnn']['b'])
        # apply nonlinear function
        r_c = tf.nn.tanh(r_c)

        # max-over-time pooling
        # shape = [B * T_s * T_w, T_c + h - 1, D_r, 1]
        r_c = tf.expand_dims(r_c, 3)
        # max-over-time-pooling, shape = [B * T_s * T_w, 1, D_r, 1]
        r_c = tf.nn.max_pool(r_c,
                             ksize=[1, T_c + h - 1, 1, 1],
                             strides=[1, 1, 1, 1],
                             padding='VALID',
                             )
        # shape = [B * T_s * T_w, D_r]
        r_c = tf.squeeze(r_c)
        #  shape = [B, T_s, T_w, D_r]
        r_c = tf.reshape(r_c, [2 * self.B, T_s, T_w, D_r])

        return r_c

    #############################################
    # BiLSTM layer for words-to-sentence encoding
    #############################################
    def bilstm_layer_ws(self, e_w_f, N_w):

        D_w = self.hyper_parameters['D_w']
        D_r = self.hyper_parameters['D_r']
        D_s = self.hyper_parameters['D_s']
        T_w = self.hyper_parameters['T_w']
        T_s = self.hyper_parameters['T_s']

        # reshape N_w, shape = [B * T_s]
        N_w = tf.reshape(N_w, shape=[2 * self.B * T_s])
        # reshape input word embeddings, shape = [B * T_s, T_w, D_w + D_r]
        e_w_f = tf.reshape(e_w_f, shape=[2 * self.B * T_s, T_w, D_w + D_r])
        # reverse input sentences
        e_w_b = tf.reverse_sequence(e_w_f, seq_lengths=N_w, seq_axis=1)

        h_0_f = tf.zeros(shape=[2 * self.B * T_s, D_s], dtype=tf.float32)
        h_0_b = tf.zeros(shape=[2 * self.B * T_s, D_s], dtype=tf.float32)
        c_0_f = tf.zeros(shape=[2 * self.B * T_s, D_s], dtype=tf.float32)
        c_0_b = tf.zeros(shape=[2 * self.B * T_s, D_s], dtype=tf.float32)

        t_0 = tf.constant(0, dtype=tf.int32)

        N_w_t = tf.tile(tf.expand_dims(N_w, axis=1), tf.constant([1, T_w], tf.int32))

        states = tf.scan(self.bilstm_cell_ws,
                         [tf.transpose(e_w_f, perm=[1, 0, 2]),
                          tf.transpose(e_w_b, perm=[1, 0, 2]),
                          tf.transpose(N_w_t, perm=[1, 0])],
                         initializer=[h_0_f, h_0_b, c_0_f, c_0_b, t_0],
                         name='lstm_ws_layer',
                         )

        h_f = states[0]
        h_b = states[1]
        h_f = tf.transpose(h_f, perm=[1, 0, 2])
        h_b = tf.transpose(h_b, perm=[1, 0, 2])
        # reverse again backward state
        h_b = tf.reverse_sequence(h_b, seq_lengths=N_w, seq_axis=1)

        # concatenate hidden states, shape=[2 * B * T_s, T_w, 2 * D_s]
        h = tf.concat([h_f, h_b], axis=2)
        # reshape input word embeddings, shape = [2 * B, T_s, T_w, 2 * D_s]
        h = tf.reshape(h, shape=[2 * self.B, T_s, T_w, 2 * D_s])

        return h

    ############################################
    # BiLSTM cell for words-to-sentence encoding
    ############################################
    def bilstm_cell_ws(self, prev, input):

        # input parameters
        h_prev_f = prev[0]
        h_prev_b = prev[1]
        c_prev_f = prev[2]
        c_prev_b = prev[3]
        t = prev[4]

        e_w_f = input[0]
        e_w_b = input[1]
        N_w = input[2]

        # compute next forward states
        h_next_f, c_next_f = self.lstm_cell(e_w_f, h_prev_f, c_prev_f,
                                            self.theta['lstm_ws_forward'],
                                            self.dropout['lstm_ws_forward'],
                                            )
        # compute next backward states
        h_next_b, c_next_b = self.lstm_cell(e_w_b, h_prev_b, c_prev_b,
                                            self.theta['lstm_ws_backward'],
                                            self.dropout['lstm_ws_backward'],
                                            )

        # t < T
        condition = tf.less(t, N_w)

        # copy-through states if t > T
        h_next_f = tf.where(condition, h_next_f, h_prev_f)
        c_next_f = tf.where(condition, c_next_f, c_prev_f)
        h_next_b = tf.where(condition, h_next_b, h_prev_b)
        c_next_b = tf.where(condition, c_next_b, c_prev_b)

        return [h_next_f, h_next_b, c_next_f, c_next_b, tf.add(t, 1)]

    #################################################
    # BiLSTM layer for sentences-to-document encoding
    #################################################
    def bilstm_layer_sd(self, e_s_f, N_s):

        D_d = self.hyper_parameters['D_d']
        T_s = self.hyper_parameters['T_s']

        # reverse input sentences
        e_s_b = tf.reverse_sequence(e_s_f, seq_lengths=N_s, seq_axis=1)

        h_0_f = tf.zeros(shape=[2 * self.B, D_d], dtype=tf.float32)
        h_0_b = tf.zeros(shape=[2 * self.B, D_d], dtype=tf.float32)
        c_0_f = tf.zeros(shape=[2 * self.B, D_d], dtype=tf.float32)
        c_0_b = tf.zeros(shape=[2 * self.B, D_d], dtype=tf.float32)

        t_0 = tf.constant(0, dtype=tf.int32)

        N_s_t = tf.tile(tf.expand_dims(N_s, axis=1), tf.constant([1, T_s], tf.int32))

        states = tf.scan(self.bilstm_cell_sd,
                         [tf.transpose(e_s_f, perm=[1, 0, 2]),
                          tf.transpose(e_s_b, perm=[1, 0, 2]),
                          tf.transpose(N_s_t, perm=[1, 0])],
                         initializer=[h_0_f, h_0_b, c_0_f, c_0_b, t_0],
                         name='lstm_sd_layer',
                         )

        h_f = states[0]
        h_b = states[1]
        h_f = tf.transpose(h_f, perm=[1, 0, 2])
        h_b = tf.transpose(h_b, perm=[1, 0, 2])
        # reverse again backward state
        h_b = tf.reverse_sequence(h_b, seq_lengths=N_s, seq_axis=1)

        # concatenate hidden states, shape=[2 * B, T_s, 2 * D_d]
        h = tf.concat([h_f, h_b], axis=2)

        return h

    ################################################
    # BiLSTM cell for sentences-to-document encoding
    ################################################
    def bilstm_cell_sd(self, prev, input):

        # input parameters
        h_prev_f = prev[0]
        h_prev_b = prev[1]
        c_prev_f = prev[2]
        c_prev_b = prev[3]
        t = prev[4]

        e_s_f = input[0]
        e_s_b = input[1]
        N_s = input[2]

        # compute next forward states
        h_next_f, c_next_f = self.lstm_cell(e_s_f, h_prev_f, c_prev_f,
                                            self.theta['lstm_sd_forward'],
                                            self.dropout['lstm_sd_forward'],
                                            )
        # compute next backward states
        h_next_b, c_next_b = self.lstm_cell(e_s_b, h_prev_b, c_prev_b,
                                            self.theta['lstm_sd_backward'],
                                            self.dropout['lstm_sd_backward'],
                                            )

        # t < T
        condition = tf.less(t, N_s)

        # copy-through states if t > T
        h_next_f = tf.where(condition, h_next_f, h_prev_f)
        c_next_f = tf.where(condition, c_next_f, c_prev_f)
        h_next_b = tf.where(condition, h_next_b, h_prev_b)
        c_next_b = tf.where(condition, c_next_b, c_prev_b)

        return [h_next_f, h_next_b, c_next_f, c_next_b, tf.add(t, 1)]

    ##################
    # single LSTM cell
    ##################
    def lstm_cell(self, e_w, h_prev, c_prev, params, dropout):

        dropout_x = tf.concat([dropout['x'],
                               dropout['x']],
                              axis=0)
        dropout_h = tf.concat([dropout['h'],
                               dropout['h']],
                              axis=0)

        W_i = params['W_i']
        U_i = params['U_i']
        b_i = params['b_i']
        W_f = params['W_f']
        U_f = params['U_f']
        b_f = params['b_f']
        W_o = params['W_o']
        U_o = params['U_o']
        b_o = params['b_o']
        W_c = params['W_c']
        U_c = params['U_c']
        b_c = params['b_c']

        is_training = self.placeholders['is_training']

        e_w = tf.cond(tf.equal(is_training, tf.constant(True)),
                      lambda: tf.multiply(dropout_x, e_w),
                      lambda: e_w,
                      )
        h_prev = tf.cond(tf.equal(is_training, tf.constant(True)),
                         lambda: tf.multiply(dropout_h, h_prev),
                         lambda: h_prev,
                         )
        # forget
        i_t = tf.sigmoid(tf.matmul(e_w, W_i) + tf.matmul(h_prev, U_i) + b_i)
        # input
        f_t = tf.sigmoid(tf.matmul(e_w, W_f) + tf.matmul(h_prev, U_f) + b_f)
        # new memory
        c_tilde = tf.tanh(tf.matmul(e_w, W_c) + tf.matmul(h_prev, U_c) + b_c)
        # final memory
        c_next = tf.multiply(i_t, c_tilde) + tf.multiply(f_t, c_prev)
        # output
        o_t = tf.sigmoid(tf.matmul(e_w, W_o) + tf.matmul(h_prev, U_o) + b_o)
        # next hidden state
        h_next = tf.multiply(o_t, tf.tanh(c_next))

        return h_next, c_next

    ################################################
    # attention layer for words-to-sentence encoding
    ################################################
    def att_layer_ws(self, h_w, N_w):

        D_s = self.hyper_parameters['D_s']
        T_w = self.hyper_parameters['T_w']
        T_s = self.hyper_parameters['T_s']

        # prepare "siamese" dropout
        is_training = self.placeholders['is_training']
        dropout_Wb = tf.concat([self.dropout['att_ws']['Wb'],
                                self.dropout['att_ws']['Wb']],
                               axis=0)
        dropout_v = tf.concat([self.dropout['att_ws']['v'],
                               self.dropout['att_ws']['v']],
                              axis=0)

        # apply dropout, shape=[2 * B, T_s, T_w, 2 * D_s]
        scores = tf.cond(tf.equal(is_training, tf.constant(True)),
                         lambda: tf.multiply(dropout_Wb, h_w),
                         lambda: h_w,
                         )
        # shape=[2 * B * T_s * T_w, 2 * D_s]
        scores = tf.reshape(scores, shape=[2 * self.B * T_s * T_w, 2 * D_s])
        scores = tf.nn.tanh(tf.nn.xw_plus_b(scores,
                                            self.theta['att_ws']['W'],
                                            self.theta['att_ws']['b']))
        # shape=[2 * B * T_s, T_w, D_s]
        scores = tf.reshape(scores, shape=[2 * self.B * T_s, T_w, 2 * D_s])
        # apply dropout
        scores = tf.cond(tf.equal(is_training, tf.constant(True)),
                         lambda: tf.multiply(dropout_v, scores),
                         lambda: scores,
                         )
        # shape=[2 * B * T_s * T_w, 2 * D_s]
        scores = tf.reshape(scores, shape=[2 * self.B * T_s * T_w, 2 * D_s])

        # shape=[2 * B * T_s * T_w, 1]
        scores = tf.matmul(scores, self.theta['att_ws']['v'])
        # shape=[2 * B, T_s, T_w]
        scores = tf.reshape(scores, shape=[2 * self.B, T_s, T_w])

        # binary mask, shape = [2 * B, T_s, T_w]
        mask = tf.sequence_mask(N_w, maxlen=T_w, dtype=tf.float32)
        mask = (1.0 - mask) * -5000.0

        # shape = [2 * B, T_s, T_w]
        scores = scores + mask
        scores = tf.nn.softmax(scores, axis=2)

        # expand to shape=[B, T_s, T_w, 1]
        alpha = tf.expand_dims(scores, axis=3)
        # fill up to shape=[B, T_s, T_w, D_s]
        alpha = tf.tile(alpha, tf.stack([1, 1, 1, 2 * D_s]))
        # combine to get sentence representations, shape=[B, T_s, 2 * D_s]
        e_s = tf.reduce_sum(tf.multiply(alpha, h_w), axis=2, keepdims=False)

        return e_s

    ####################################################
    # attention layer for sentences-to-docuemnt encoding
    ####################################################
    def att_layer_sd(self, h_s, N_s):

        D_d = self.hyper_parameters['D_d']
        T_s = self.hyper_parameters['T_s']

        # prepare "siamese" dropout
        is_training = self.placeholders['is_training']
        dropout_Wb = tf.concat([self.dropout['att_sd']['Wb'],
                                self.dropout['att_sd']['Wb']],
                               axis=0)
        dropout_v = tf.concat([self.dropout['att_sd']['v'],
                               self.dropout['att_sd']['v']],
                              axis=0)

        # apply dropout, shape=[2 * B, T_s, 2 * D_d]
        scores = tf.cond(tf.equal(is_training, tf.constant(True)),
                         lambda: tf.multiply(dropout_Wb, h_s),
                         lambda: h_s,
                         )
        # shape=[2 * B * T_s, 2 * D_d]
        scores = tf.reshape(scores, shape=[2 * self.B * T_s, 2 * D_d])
        scores = tf.nn.tanh(tf.nn.xw_plus_b(scores,
                                            self.theta['att_sd']['W'],
                                            self.theta['att_sd']['b']))
        # shape=[2 * B, T_s, 2 * D_d]
        scores = tf.reshape(scores, shape=[2 * self.B, T_s, 2 * D_d])

        # apply dropout
        scores = tf.cond(tf.equal(is_training, tf.constant(True)),
                         lambda: tf.multiply(dropout_v, scores),
                         lambda: scores,
                         )
        # shape=[2 * B * T_s, 2 * D_d]
        scores = tf.reshape(scores, shape=[2 * self.B * T_s, 2 * D_d])

        # shape=[2 * B * T_s, 1]
        scores = tf.matmul(scores, self.theta['att_sd']['v'])
        # shape=[2 * B, T_s]
        scores = tf.reshape(scores, shape=[2 * self.B, T_s])

        # binary mask, shape = [2 * B, T_s]
        mask = tf.sequence_mask(N_s, maxlen=T_s, dtype=tf.float32)
        mask = (1.0 - mask) * -5000.0

        # shape = [2 * B, T_s]
        scores = scores + mask
        scores = tf.nn.softmax(scores, axis=1)

        # expand to shape=[B, T_s, 1]
        alpha = tf.expand_dims(scores, axis=2)
        # fill up to shape=[B, T_s, 2 * D_d]
        alpha = tf.tile(alpha, tf.stack([1, 1, 2 * D_d]))
        # combine to get doc representations, shape=[B, 2 * D_d]
        e_d = tf.reduce_sum(tf.multiply(alpha, h_s), axis=1, keepdims=False)

        return e_d

    #########
    # dropout
    #########
    def make_dropout_mask(self, shape, keep_prob):
        keep_prob = tf.convert_to_tensor(keep_prob, dtype=tf.float32)
        random_tensor = keep_prob + tf.random_uniform(shape, dtype=tf.float32)
        binary_tensor = tf.floor(random_tensor)
        dropout_mask = tf.divide(binary_tensor, keep_prob)
        return dropout_mask

    def initialize_dropout(self):

        D_c = self.hyper_parameters['D_c']
        D_r = self.hyper_parameters['D_r']
        D_w = self.hyper_parameters['D_w']
        D_s = self.hyper_parameters['D_s']
        D_d = self.hyper_parameters['D_d']
        T_c = self.hyper_parameters['T_c']
        T_w = self.hyper_parameters['T_w']
        T_s = self.hyper_parameters['T_s']
        D_lev = self.hyper_parameters['D_lev']

        dropout = {}

        with tf.variable_scope('dropout_cnn'):
            dropout['cnn'] = self.make_dropout_mask(shape=[self.B * T_s * T_w, T_c, D_c],
                                                    keep_prob=self.hyper_parameters['keep_prob_cnn'],
                                                    )
        with tf.variable_scope('dropout_lstm_ws_forward'):
            dropout['lstm_ws_forward'] = {}
            dropout['lstm_ws_forward']['x'] = self.make_dropout_mask(shape=[self.B * T_s, D_w + D_r],
                                                                     keep_prob=self.hyper_parameters['keep_prob_lstm'],
                                                                     )
            dropout['lstm_ws_forward']['h'] = self.make_dropout_mask(shape=[self.B * T_s, D_s],
                                                                     keep_prob=self.hyper_parameters['keep_prob_lstm'],
                                                                     )
        with tf.variable_scope('dropout_lstm_ws_backward'):
            dropout['lstm_ws_backward'] = {}
            dropout['lstm_ws_backward']['x'] = self.make_dropout_mask(shape=[self.B * T_s, D_w + D_r],
                                                                      keep_prob=self.hyper_parameters['keep_prob_lstm'],
                                                                      )
            dropout['lstm_ws_backward']['h'] = self.make_dropout_mask(shape=[self.B * T_s, D_s],
                                                                      keep_prob=self.hyper_parameters['keep_prob_lstm'],
                                                                      )
        with tf.variable_scope('dropout_lstm_sd_forward'):
            dropout['lstm_sd_forward'] = {}
            dropout['lstm_sd_forward']['x'] = self.make_dropout_mask(shape=[self.B, 2 * D_s],
                                                                     keep_prob=self.hyper_parameters['keep_prob_lstm'],
                                                                     )
            dropout['lstm_sd_forward']['h'] = self.make_dropout_mask(shape=[self.B, D_d],
                                                                     keep_prob=self.hyper_parameters['keep_prob_lstm'],
                                                                     )
        with tf.variable_scope('dropout_lstm_sd_backward'):
            dropout['lstm_sd_backward'] = {}
            dropout['lstm_sd_backward']['x'] = self.make_dropout_mask(shape=[self.B, 2 * D_s],
                                                                      keep_prob=self.hyper_parameters['keep_prob_lstm'],
                                                                      )
            dropout['lstm_sd_backward']['h'] = self.make_dropout_mask(shape=[self.B, D_d],
                                                                      keep_prob=self.hyper_parameters['keep_prob_lstm'],
                                                                      )
        with tf.variable_scope('dropout_att_ws'):
            dropout['att_ws'] = {}
            dropout['att_ws']['Wb'] = self.make_dropout_mask(shape=[self.B, T_s, 1, 2 * D_s],
                                                             keep_prob=self.hyper_parameters['keep_prob_att'],
                                                             )
            dropout['att_ws']['v'] = self.make_dropout_mask(shape=[self.B * T_s, 1, 2 * D_s],
                                                            keep_prob=self.hyper_parameters['keep_prob_att'],
                                                            )
        with tf.variable_scope('dropout_att_sd'):
            dropout['att_sd'] = {}
            dropout['att_sd']['Wb'] = self.make_dropout_mask(shape=[self.B, 1, 2 * D_d],
                                                             keep_prob=self.hyper_parameters['keep_prob_att'],
                                                             )
            dropout['att_sd']['v'] = self.make_dropout_mask(shape=[self.B, 1, 2 * D_d],
                                                            keep_prob=self.hyper_parameters['keep_prob_att'],
                                                            )
        with tf.variable_scope('dropout_metric_a'):
            dropout['metric_a'] = {}
            dropout['metric_a']['metric_1'] = self.make_dropout_mask(shape=[self.B, 2 * D_d],
                                                              keep_prob=self.hyper_parameters['keep_prob_metric'],
                                                              )
        with tf.variable_scope('dropout_metric_f'):
            dropout['metric_f'] = {}
            dropout['metric_f']['metric_1'] = self.make_dropout_mask(shape=[self.B, 2 * D_d],
                                                              keep_prob=self.hyper_parameters['keep_prob_metric'],
                                                              )

        with tf.variable_scope('dropout_bfs_a'):
            dropout['bfs_a'] = {}
            dropout['bfs_a']['bfs_1'] = self.make_dropout_mask(shape=[self.B, D_lev],
                                                               keep_prob=self.hyper_parameters['keep_prob_bfs'],
                                                               )
        with tf.variable_scope('dropout_bfs_f'):
            dropout['bfs_f'] = {}
            dropout['bfs_f']['bfs_1'] = self.make_dropout_mask(shape=[self.B, D_lev],
                                                               keep_prob=self.hyper_parameters['keep_prob_bfs'],
                                                               )
        with tf.variable_scope('dropout_ual_a'):
            dropout['ual_a'] = {}
            dropout['ual_a']['ual_1'] = self.make_dropout_mask(shape=[self.B, D_lev],
                                                                 keep_prob=self.hyper_parameters['keep_prob_ual'],
                                                                 )
            dropout['ual_a']['ual_2'] = self.make_dropout_mask(shape=[self.B, 2*D_lev],
                                                                 keep_prob=self.hyper_parameters['keep_prob_ual'],
                                                                 )
        return dropout

    def initialize_parameters(self):

        D_c = self.hyper_parameters['D_c']
        D_r = self.hyper_parameters['D_r']
        h = self.hyper_parameters['w']
        D_w = self.hyper_parameters['D_w']
        D_s = self.hyper_parameters['D_s']
        D_d = self.hyper_parameters['D_d']
        D_lev = self.hyper_parameters['D_lev']
        D_plda = self.hyper_parameters['D_plda']

        theta = {}

        with tf.variable_scope('theta_cnn'):
            theta['cnn'] = self.initialize_cnn(D_c, D_r, h)

        with tf.variable_scope('theta_lstm_ws_forward'):
            theta['lstm_ws_forward'] = self.initialize_lstm(D_w + D_r, D_s)
        with tf.variable_scope('theta_lstm_ws_backward'):
            theta['lstm_ws_backward'] = self.initialize_lstm(D_w + D_r, D_s)

        with tf.variable_scope('theta_lstm_sd_forward'):
            theta['lstm_sd_forward'] = self.initialize_lstm(2 * D_s, D_d)
        with tf.variable_scope('theta_lstm_sd_backward'):
            theta['lstm_sd_backward'] = self.initialize_lstm(2 * D_s, D_d)

        with tf.variable_scope('theta_att_ws'):
            theta['att_ws'] = self.initialize_att(2 * D_s, 2 * D_s)
        with tf.variable_scope('theta_att_sd'):
            theta['att_sd'] = self.initialize_att(2 * D_d, 2 * D_d)

        with tf.variable_scope('theta_metric_a'):
            theta['metric_a'] = self.initialize_mlp(2 * D_d, D_lev)
        with tf.variable_scope('theta_metric_f'):
            theta['metric_f'] = self.initialize_mlp(2 * D_d, D_lev)

        with tf.variable_scope('theta_bfs_a_1'):
            theta['bfs_a_1'] = self.initialize_mlp(D_lev, D_plda)
        with tf.variable_scope('theta_bfs_a_2'):
            theta['bfs_a_2'] = self.initialize_bfs(D_plda)

        with tf.variable_scope('theta_bfs_f_1'):
            theta['bfs_f_1'] = self.initialize_mlp(D_lev, D_plda)
        with tf.variable_scope('theta_bfs_f_2'):
            theta['bfs_f_2'] = self.initialize_bfs(D_plda)

        with tf.variable_scope('theta_ual_a'):
            theta['ual_a'] = self.initialize_ual(D_lev)

        with tf.variable_scope('theta_dml_loss_a'):
            theta['loss_dml_a'] = self.initialize_theta_loss()

        return theta

    def initialize_theta_loss(self):
        theta = {'alpha': tf.get_variable(name='loss_alpha',
                                          shape=[1],
                                          initializer=tf.constant_initializer(0.09),
                                          trainable=self.hyper_parameters["train_kernel"],
                                          dtype=tf.float32,
                                          ),
                 'beta': tf.get_variable(name='loss_beta',
                                         shape=[1],
                                         initializer=tf.constant_initializer(3.0 / 2.0),
                                         trainable=self.hyper_parameters["train_kernel"],
                                         dtype=tf.float32,
                                         ),
                 }

        return theta

    def initialize_ual(self, D):
        theta = {'W1': tf.get_variable(name='W1',
                                       shape=[D, 2*D],
                                       initializer=tf.contrib.layers.xavier_initializer(),
                                       trainable=True,
                                       dtype=tf.float32,
                                       ),
                 'b1': tf.get_variable(name='b1',
                                       shape=[2*D],
                                       initializer=tf.constant_initializer(0.0),
                                       trainable=True,
                                       dtype=tf.float32,
                                       ),
                 'W2': tf.get_variable(name='W2',
                                       shape=[2*D, 4],
                                       initializer=tf.contrib.layers.xavier_initializer(),
                                       trainable=True,
                                       dtype=tf.float32,
                                       ),
                 'b2': tf.get_variable(name='b2',
                                       shape=[4],
                                       initializer=tf.constant_initializer(0.0),
                                       trainable=True,
                                       dtype=tf.float32,
                                       ),

                 }

        return theta

    def initialize_bfs(self, D):
        m_W = 0.02 * np.random.rand(D, D)
        m_B = 0.02 * np.random.rand(D, D)
        np.fill_diagonal(m_W, np.log(1.02))
        np.fill_diagonal(m_B, np.log(0.98))
        theta = {'W': tf.Variable(np.array(m_W, dtype=np.float32), name='W', trainable=True),
                 'B': tf.Variable(np.array(m_B, dtype=np.float32), name='B', trainable=True),
                 'mu': tf.get_variable(name='mu',
                                       shape=[D, 1],
                                       initializer=tf.initializers.random_uniform(minval=-0.0001, maxval=0.0001),
                                       trainable=True,
                                       ),
                 }
        return theta

    def initialize_mlp(self, D_in, D_out):
        r = 0.4
        theta = {'W': tf.get_variable('W',
                                      shape=[D_in, D_out],
                                      initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                      trainable=True,
                                      dtype=tf.float32,
                                      ),
                 'b': tf.get_variable('b',
                                      shape=[D_out],
                                      initializer=tf.constant_initializer(0.0),
                                      trainable=True,
                                      dtype=tf.float32,
                                      ),
                 }
        return theta

    def initialize_cnn(self, D_in, D_out, h):
        r = 0.1
        theta = {'W': tf.get_variable(name='W',
                                      shape=[h, D_in, D_out],
                                      initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                      trainable=True,
                                      dtype=tf.float32,
                                      ),
                 'b': tf.get_variable(name='b',
                                      shape=[D_out],
                                      initializer=tf.constant_initializer(0.0),
                                      trainable=True,
                                      dtype=tf.float32,
                                      ),
                 }

        return theta

    def initialize_att(self, D_in, D_out):
        r = 0.03
        theta = {'W': tf.get_variable('W_a',
                                      shape=[D_in, D_out],
                                      initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                      trainable=True,
                                      dtype=tf.float32,
                                      ),
                 'v': tf.get_variable('v_a', shape=[D_out, 1],
                                      initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                      trainable=True,
                                      dtype=tf.float32,
                                      ),
                 'b': tf.get_variable('b_a', shape=[D_out],
                                      initializer=tf.constant_initializer(0.0),
                                      trainable=True,
                                      dtype=tf.float32,
                                      ),
                 }

        return theta

    def initialize_lstm(self, D_in, D_out):
        r = 0.05
        theta = {'W_i': tf.get_variable('W_i', shape=[D_in, D_out],
                                        initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                        trainable=True,
                                        dtype=tf.float32,
                                        ),
                 'U_i': tf.get_variable('U_i', shape=[D_out, D_out],
                                        initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                        trainable=True,
                                        dtype=tf.float32,
                                        ),
                 'b_i': tf.get_variable('b_i', shape=[1, D_out],
                                        initializer=tf.constant_initializer(0.0),
                                        trainable=True,
                                        dtype=tf.float32,
                                        ),
                 'W_f': tf.get_variable('W_f', shape=[D_in, D_out],
                                        initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                        trainable=True,
                                        dtype=tf.float32,
                                        ),
                 'U_f': tf.get_variable('U_f', shape=[D_out, D_out],
                                        initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                        trainable=True,
                                        dtype=tf.float32,
                                        ),
                 'b_f': tf.get_variable('b_f', shape=[1, D_out],
                                        initializer=tf.constant_initializer(2.5),
                                        trainable=True,
                                        dtype=tf.float32,
                                        ),
                 'W_c': tf.get_variable('W_c', shape=[D_in, D_out],
                                        initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                        trainable=True,
                                        dtype=tf.float32,
                                        ),
                 'U_c': tf.get_variable('U_c', shape=[D_out, D_out],
                                        initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                        trainable=True,
                                        dtype=tf.float32,
                                        ),
                 'b_c': tf.get_variable('b_c', shape=[1, D_out],
                                        initializer=tf.constant_initializer(0.0),
                                        trainable=True,
                                        dtype=tf.float32,
                                        ),
                 'W_o': tf.get_variable('W_o', shape=[D_in, D_out],
                                        initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                        trainable=True,
                                        dtype=tf.float32,
                                        ),
                 'U_o': tf.get_variable('U_o', shape=[D_out, D_out],
                                        initializer=tf.initializers.random_uniform(minval=-r, maxval=r),
                                        trainable=True,
                                        dtype=tf.float32,
                                        ),
                 'b_o': tf.get_variable('b_o', shape=[1, D_out],
                                        initializer=tf.constant_initializer(1.0),
                                        trainable=True,
                                        dtype=tf.float32,
                                        )
                 }
        return theta

    #############################
    # update weights of the model
    #############################
    def update_model(self, x_w_L, x_w_R, x_c_L, x_c_R, labels_a, labels_f, N_w_L, N_w_R, N_s_L, N_s_R, lr):

        feed_dict = {self.placeholders['x_w_L']: x_w_L,
                     self.placeholders['x_w_R']: x_w_R,
                     self.placeholders['x_c_L']: x_c_L,
                     self.placeholders['x_c_R']: x_c_R,
                     self.placeholders['labels_a']: labels_a,
                     self.placeholders['labels_f']: labels_f,
                     self.placeholders['N_w_L']: N_w_L,
                     self.placeholders['N_w_R']: N_w_R,
                     self.placeholders['N_s_L']: N_s_L,
                     self.placeholders['N_s_R']: N_s_R,
                     self.placeholders['is_training']: True,
                     self.placeholders['lr']: lr,
                     }
        _, loss, \
        pred_dml_a, pred_dml_f, \
        pred_bfs_a, pred_bfs_f, \
        pred_ual_a, conf_matrix_a, \
        H_W_a, H_B_a, H_W_f, H_B_f \
            = self.sess.run([self.optimizer, self.loss,
                             self.pred_dml_a, self.pred_dml_f,
                             self.pred_bfs_a, self.pred_bfs_f,
                             self.pred_ual_a, self.conf_matrix_a,
                             self.H_W_a, self.H_B_a, self.H_W_f, self.H_B_f,
                             ], feed_dict=feed_dict)

        # execute label computation function
        labels_dml_a = self.compute_labels(pred_dml_a)
        labels_bfs_a = self.compute_labels(pred_bfs_a)
        labels_ual_a = self.compute_labels(pred_ual_a[:, 1])

        labels_dml_f = self.compute_labels(pred_dml_f)
        labels_bfs_f = self.compute_labels(pred_bfs_f)

        # compute values for accuracy, F1-score and c@1
        TP_dml_a, FP_dml_a, TN_dml_a, FN_dml_a = self.compute_TP_FP_TN_FN(labels_a, labels_dml_a)
        TP_bfs_a, FP_bfs_a, TN_bfs_a, FN_bfs_a = self.compute_TP_FP_TN_FN(labels_a, labels_bfs_a)
        TP_ual_a, FP_ual_a, TN_ual_a, FN_ual_a = self.compute_TP_FP_TN_FN(labels_a, labels_ual_a)

        TP_dml_f, FP_dml_f, TN_dml_f, FN_dml_f = self.compute_TP_FP_TN_FN(labels_f, labels_dml_f)
        TP_bfs_f, FP_bfs_f, TN_bfs_f, FN_bfs_f = self.compute_TP_FP_TN_FN(labels_f, labels_bfs_f)

        return loss, \
               TP_dml_a, FP_dml_a, TN_dml_a, FN_dml_a, \
               TP_bfs_a, FP_bfs_a, TN_bfs_a, FN_bfs_a, \
               TP_ual_a, FP_ual_a, TN_ual_a, FN_ual_a, \
               TP_dml_f, FP_dml_f, TN_dml_f, FN_dml_f, \
               TP_bfs_f, FP_bfs_f, TN_bfs_f, FN_bfs_f, \
               H_W_a, H_B_a, H_W_f, H_B_f, \
               conf_matrix_a[0, :, :]

    @staticmethod
    def compute_confidence(pred, labels_hat):
        confidences = pred.copy()
        confidences[labels_hat == 0] = 1.0 - confidences[labels_hat == 0]
        return confidences

    ################
    # evaluate model
    ################
    def evaluate_model(self, docs_L, docs_R, labels, batch_size):

        num_batches = ceil(len(labels) / batch_size)

        TP_dml, FP_dml, TN_dml, FN_dml = 0, 0, 0, 0
        TP_bfs, FP_bfs, TN_bfs, FN_bfs = 0, 0, 0, 0
        TP_ual, FP_ual, TN_ual, FN_ual = 0, 0, 0, 0

        pred_dml, pred_bfs, pred_ual = [], [], []

        for i in range(num_batches):

            # get next batch
            docs_L_i, docs_R_i, labels_i, _ = self.next_batch(i * batch_size,
                                                              (i + 1) * batch_size,
                                                              docs_L,
                                                              docs_R,
                                                              labels,
                                                              labels,
                                                              )
            B = len(labels_i)

            if B > 0:
                # word/character embeddings
                x_w_L, N_w_L, N_s_L, x_c_L = self.doc2mat(docs_L_i)
                x_w_R, N_w_R, N_s_R, x_c_R = self.doc2mat(docs_R_i)

                # accuracy for training set
                curr_TP_dml, curr_FP_dml, curr_TN_dml, curr_FN_dml, \
                curr_TP_bfs, curr_FP_bfs, curr_TN_bfs, curr_FN_bfs, \
                curr_TP_ual, curr_FP_ual, curr_TN_ual, curr_FN_ual, \
                curr_pred_dml, curr_pred_bfs, curr_pred_ual \
                    = self.compute_eval_measures(x_w_L=x_w_L,
                                                 x_w_R=x_w_R,
                                                 x_c_L=x_c_L,
                                                 x_c_R=x_c_R,
                                                 labels=np.array(labels_i).reshape((B, 1)),
                                                 N_w_L=N_w_L,
                                                 N_w_R=N_w_R,
                                                 N_s_L=N_s_L,
                                                 N_s_R=N_s_R,
                                                 )
                TP_dml += curr_TP_dml
                FP_dml += curr_FP_dml
                TN_dml += curr_TN_dml
                FN_dml += curr_FN_dml

                TP_bfs += curr_TP_bfs
                FP_bfs += curr_FP_bfs
                TN_bfs += curr_TN_bfs
                FN_bfs += curr_FN_bfs

                TP_ual += curr_TP_ual
                FP_ual += curr_FP_ual
                TN_ual += curr_TN_ual
                FN_ual += curr_FN_ual

                pred_dml.extend(curr_pred_dml)
                pred_bfs.extend(curr_pred_bfs)
                pred_ual.extend(curr_pred_ual)

        labels = np.array(labels).reshape(-1)

        pred_dml = np.array(pred_dml).reshape(-1)
        labels_dml = self.compute_labels(pred_dml).reshape(-1)

        pred_bfs = np.array(pred_bfs).reshape(-1)
        labels_bfs = self.compute_labels(pred_bfs).reshape(-1)

        pred_ual = np.array(pred_ual).reshape(-1)
        labels_ual = self.compute_labels(pred_ual).reshape(-1)

        acc_dml = self.compute_accuracy(TP_dml, FP_dml, TN_dml, FN_dml)
        acc_bfs = self.compute_accuracy(TP_bfs, FP_bfs, TN_bfs, FN_bfs)
        acc_ual = self.compute_accuracy(TP_ual, FP_ual, TN_ual, FN_ual)

        # compute expected calibration error and maximum cal. error
        confidence_dml = self.compute_confidence(pred_dml, labels_dml)
        calibration_dml = compute_calibration(true_labels=labels,
                                              pred_labels=labels_dml,
                                              confidences=confidence_dml,
                                              )
        confidence_bfs = self.compute_confidence(pred_bfs, labels_bfs)
        calibration_bfs = compute_calibration(true_labels=labels,
                                              pred_labels=labels_bfs,
                                              confidences=confidence_bfs,
                                              )
        confidence_ual = self.compute_confidence(pred_ual, labels_ual)
        calibration_ual = compute_calibration(true_labels=labels,
                                              pred_labels=labels_ual,
                                              confidences=confidence_ual,
                                              )

        pan_dml = evaluate_all(pred_y=pred_dml, true_y=labels)
        pan_bfs = evaluate_all(pred_y=pred_bfs, true_y=labels)
        pan_ual = evaluate_all(pred_y=pred_ual, true_y=labels)

        return acc_dml, acc_bfs, acc_ual, \
               pan_dml, pan_bfs, pan_ual, \
               calibration_dml, calibration_bfs, calibration_ual

    ###############################################
    # evaluate model 'AdHominem' for a single batch
    ###############################################
    def compute_eval_measures(self, x_w_L, x_w_R, x_c_L, x_c_R, labels, N_w_L, N_w_R, N_s_L, N_s_R):

        # compute distances
        pred_dml, pred_bfs, pred_ual = self.sess.run([self.pred_dml_a, self.pred_bfs_a, self.pred_ual_a],
                                                     feed_dict={self.placeholders['x_w_L']: x_w_L,
                                                                self.placeholders['x_w_R']: x_w_R,
                                                                self.placeholders['x_c_L']: x_c_L,
                                                                self.placeholders['x_c_R']: x_c_R,
                                                                self.placeholders['N_w_L']: N_w_L,
                                                                self.placeholders['N_w_R']: N_w_R,
                                                                self.placeholders['N_s_L']: N_s_L,
                                                                self.placeholders['N_s_R']: N_s_R,
                                                                self.placeholders['is_training']: False,
                                                                })

        # execute label computation function
        labels_dml = self.compute_labels(pred_dml)
        labels_bfs = self.compute_labels(pred_bfs)
        labels_ual = self.compute_labels(pred_ual[:, 1])

        # compute values for accuracy, F1-score and c@1
        TP_dml, FP_dml, TN_dml, FN_dml = self.compute_TP_FP_TN_FN(labels, labels_dml)
        TP_bfs, FP_bfs, TN_bfs, FN_bfs = self.compute_TP_FP_TN_FN(labels, labels_bfs)
        TP_ual, FP_ual, TN_ual, FN_ual = self.compute_TP_FP_TN_FN(labels, labels_ual)

        return TP_dml, FP_dml, TN_dml, FN_dml, \
               TP_bfs, FP_bfs, TN_bfs, FN_bfs, \
               TP_ual, FP_ual, TN_ual, FN_ual, \
               pred_dml, pred_bfs, pred_ual[:, 1]

    ##########################
    # calculate TP, FP, TN, FN
    ##########################
    @staticmethod
    def compute_TP_FP_TN_FN(labels, labels_hat):

        TP, FP, TN, FN = 0, 0, 0, 0

        for i in range(len(labels_hat)):
            if labels[i] == 1 and labels_hat[i] == 1:
                TP += 1
            if labels[i] == 0 and labels_hat[i] == 1:
                FP += 1
            if labels[i] == 0 and labels_hat[i] == 0:
                TN += 1
            if labels[i] == 1 and labels_hat[i] == 0:
                FN += 1

        return TP, FP, TN, FN

    ##################
    # compute accuracy
    ##################
    @staticmethod
    def compute_accuracy(TP, FP, TN, FN):

        acc = (TP + TN) / (TP + FP + TN + FN)

        return acc

    ################
    # get next batch
    ################
    @staticmethod
    def next_batch(t_s, t_e, docs_L, docs_R, labels_a, labels_f):

        docs_L = docs_L[t_s:t_e]
        docs_R = docs_R[t_s:t_e]
        labels_a = labels_a[t_s:t_e]
        labels_f = labels_f[t_s:t_e]

        return docs_L, docs_R, labels_a, labels_f

    ##################################################
    # sliding windowing for sentence-unit construction
    ##################################################
    def sliding_window(self, doc):

        T_w = self.hyper_parameters['T_w']
        hop_length = self.hyper_parameters['hop_length']

        tokens = doc.split()
        doc_new = []
        n = 0
        while len(tokens[n:n + T_w]) > 0:
            # split sentence into tokens
            sent_new = ''
            for token in tokens[n: n + T_w]:
                sent_new += token + ' '
            # add to new doc
            doc_new.append(sent_new.strip())
            # update stepsize
            n += hop_length

        return doc_new

    ########################
    # word-to-vector mapping
    ########################
    def doc2mat(self, docs, is_training=False):

        T_c = self.hyper_parameters['T_c']
        T_w = self.hyper_parameters['T_w']
        T_s = self.hyper_parameters['T_s']
        V_c = self.hyper_parameters['V_c']
        V_w = self.hyper_parameters['V_w']

        # batch size
        B = len(docs)
        N_w = np.zeros((B, T_s), dtype=np.int32)
        N_s = np.zeros((B,), dtype=np.int32)

        # word-based tensor, shape = [B, T_s, T_w]
        x_w = np.zeros((B, T_s, T_w), dtype=np.int32)
        # character-based tensor
        x_c = np.zeros((B, T_s, T_w, T_c), dtype=np.int32)

        # current document
        for i, doc in enumerate(docs):

            # apply sliding window to construct sentence like units
            doc = self.sliding_window(doc)

            if len(doc) > T_s and is_training:
                if random.uniform(0, 1) < 0.5:
                    r = random.randint(0, len(doc) - T_s)
                    doc = doc[r:r + T_s]
            N_s[i] = len(doc[:T_s])
            # current sentence
            for j, sentence in enumerate(doc[:T_s]):
                tokens = sentence.split()
                N_w[i, j] = len(tokens)
                # current tokens
                for k, token in enumerate(tokens):
                    if token in V_w:
                        x_w[i, j, k] = V_w[token]
                    else:
                        x_w[i, j, k] = V_w['<UNK>']
                    # current character
                    for l, chr in enumerate(token[:T_c]):
                        if chr in V_c:
                            x_c[i, j, k, l] = V_c[chr]
                        else:
                            x_c[i, j, k, l] = V_c['<UNK>']

        return x_w, N_w, N_s, x_c

    ##################
    # store parameters
    ##################
    def store_parameters(self, step, dire_results):

        dir_weights = os.path.join(dire_results,
                                   'weights_adhominem',
                                   )
        if not os.path.exists(dir_weights):
            os.makedirs(dir_weights)

        parameters = {'hyper_parameters': {},
                      'theta': {},
                      'theta_E': {},
                      }

        # hyper-parameters
        for hp in self.hyper_parameters:
            parameters['hyper_parameters'][hp] = self.hyper_parameters[hp]

        # character and word embeddings
        for var in self.thetas_E.keys():
            parameters['theta_E'][var] = self.sess.run(self.thetas_E[var])

        # variables for feature extraction and plda
        for layer in self.theta.keys():
            parameters['theta'][layer] = {}
            for var in self.theta[layer].keys():
                parameters['theta'][layer][var] = self.sess.run(self.theta[layer][var])

        file = os.path.join(dir_weights, 'weights_' + str(step))
        with open(file, 'wb') as f:
            pickle.dump(parameters, f)

    ##################
    # train end-to-end
    ##################
    def train_model(self, dev_set, file_results, file_tmp, dir_results):

        # total number of epochs
        epochs = self.hyper_parameters['epochs']

        # number of batches for dev/test set
        batch_size = self.hyper_parameters['batch_size']
        batch_size_dev = self.hyper_parameters['batch_size_dev']

        # dev set
        docs_L_dev, docs_R_dev, labels_dev = dev_set

        # define learning rate
        epoch_1 = 0
        epoch_2 = self.hyper_parameters['lr_epoch']
        lr_1 = self.hyper_parameters['lr_start']
        lr_2 = self.hyper_parameters['lr_end']
        p = np.array(range(epochs))
        m = (lr_2 - lr_1) / (epoch_2 - epoch_1)
        b = lr_1
        lr = np.maximum(m * p + b, lr_2)

        ################
        # start training
        ################
        open(file_tmp, 'a').write('Preprocessing steps done, start training...')
        for epoch in range(epochs):

            # store current time
            s = str(datetime.datetime.now()).split('.')[0]
            open(file_tmp, 'a').write('\n\n' + 100 * '-' + '\n')
            open(file_tmp, 'a').write('start epoch ' + str(epoch) + ': ' + s + '\n')

            # training set
            docs_L_tr, docs_R_tr, labels_a_tr, labels_f_tr = resample_pairs_train(file_results)
            # shuffle (again)
            docs_L_tr, docs_R_tr, labels_a_tr, labels_f_tr = shuffle(docs_L_tr, docs_R_tr, labels_a_tr, labels_f_tr)

            # number of training pairs
            N_tr = len(labels_a_tr)
            # number of batches for training
            num_batches_tr = ceil(N_tr / batch_size)

            # average loss and accuracy
            loss = []
            TP_dml_a, FP_dml_a, TN_dml_a, FN_dml_a = 0, 0, 0, 0
            TP_bfs_a, FP_bfs_a, TN_bfs_a, FN_bfs_a = 0, 0, 0, 0
            TP_ual_a, FP_ual_a, TN_ual_a, FN_ual_a = 0, 0, 0, 0

            TP_dml_f, FP_dml_f, TN_dml_f, FN_dml_f = 0, 0, 0, 0
            TP_bfs_f, FP_bfs_f, TN_bfs_f, FN_bfs_f = 0, 0, 0, 0

            # loop over all batches
            for i in range(num_batches_tr):

                # get next batch
                docs_L_i, docs_R_i, labels_a_i, labels_f_i = self.next_batch(i * batch_size,
                                                                             (i + 1) * batch_size,
                                                                             docs_L_tr,
                                                                             docs_R_tr,
                                                                             labels_a_tr,
                                                                             labels_f_tr,
                                                                             )

                # current batch size
                B = len(labels_a_i)

                if B > 0:
                    # word / character embeddings
                    x_w_L, N_w_L, N_s_L, x_c_L = self.doc2mat(docs_L_i, is_training=True)
                    x_w_R, N_w_R, N_s_R, x_c_R = self.doc2mat(docs_R_i, is_training=True)

                    # update model parameters
                    curr_loss, \
                    curr_TP_dml_a, curr_FP_dml_a, curr_TN_dml_a, curr_FN_dml_a, \
                    curr_TP_bfs_a, curr_FP_bfs_a, curr_TN_bfs_a, curr_FN_bfs_a, \
                    curr_TP_ual_a, curr_FP_ual_a, curr_TN_ual_a, curr_FN_ual_a, \
                    curr_TP_dml_f, curr_FP_dml_f, curr_TN_dml_f, curr_FN_dml_f, \
                    curr_TP_bfs_f, curr_FP_bfs_f, curr_TN_bfs_f, curr_FN_bfs_f, \
                    H_W_a, H_B_a, H_W_f, H_B_f, conf_matrix_a \
                        = self.update_model(x_w_L=x_w_L,
                                            x_w_R=x_w_R,
                                            x_c_L=x_c_L,
                                            x_c_R=x_c_R,
                                            labels_a=np.array(labels_a_i).reshape((B, 1)),
                                            labels_f=np.array(labels_f_i).reshape((B, 1)),
                                            N_w_L=N_w_L,
                                            N_w_R=N_w_R,
                                            N_s_L=N_s_L,
                                            N_s_R=N_s_R,
                                            lr=lr[epoch],
                                            )

                    loss.append(curr_loss)

                    TP_dml_a += curr_TP_dml_a
                    FP_dml_a += curr_FP_dml_a
                    TN_dml_a += curr_TN_dml_a
                    FN_dml_a += curr_FN_dml_a

                    TP_bfs_a += curr_TP_bfs_a
                    FP_bfs_a += curr_FP_bfs_a
                    TN_bfs_a += curr_TN_bfs_a
                    FN_bfs_a += curr_FN_bfs_a

                    TP_ual_a += curr_TP_ual_a
                    FP_ual_a += curr_FP_ual_a
                    TN_ual_a += curr_TN_ual_a
                    FN_ual_a += curr_FN_ual_a

                    TP_dml_f += curr_TP_dml_f
                    FP_dml_f += curr_FP_dml_f
                    TN_dml_f += curr_TN_dml_f
                    FN_dml_f += curr_FN_dml_f

                    TP_bfs_f += curr_TP_bfs_f
                    FP_bfs_f += curr_FP_bfs_f
                    TN_bfs_f += curr_TN_bfs_f
                    FN_bfs_f += curr_FN_bfs_f

                    s = 'Epoch:' + str(epoch) \
                        + ', B: ' + str(round(100 * (i + 1) / num_batches_tr, 0)) \
                        + ', #P: ' + str(N_tr) \
                        + ', L: ' + str(round(float(np.mean(loss)), 3)) \
                        + ', A(dml): ' + str(round(100 * (TP_dml_a + TN_dml_a) / (TP_dml_a + FP_dml_a + TN_dml_a + FN_dml_a), 2)) \
                        + ', A(bfs): ' + str(round(100 * (TP_bfs_a + TN_bfs_a) / (TP_bfs_a + FP_bfs_a + TN_bfs_a + FN_bfs_a), 2)) \
                        + ', A(ual): ' + str(round(100 * (TP_ual_a + TN_ual_a) / (TP_ual_a + FP_ual_a + TN_ual_a + FN_ual_a), 2)) \
                        + ', A(f-dml): ' + str(round(100 * (TP_dml_f + TN_dml_f) / (TP_dml_f + FP_dml_f + TN_dml_f + FN_dml_f), 2)) \
                        + ', A(f-bfs): ' + str(round(100 * (TP_bfs_f + TN_bfs_f) / (TP_bfs_f + FP_bfs_f + TN_bfs_f + FN_bfs_f), 2)) \
                        + ', H_W(a): ' + str(round(H_W_a, 3)) \
                        + ', H_B(a): ' + str(round(H_B_a, 3)) \
                        + ', H_W(f): ' + str(round(H_W_f, 3)) \
                        + ', H_B(f): ' + str(round(H_B_f, 3)) \
                        + ', LR: ' + str(round(lr[epoch], 5)) \
                        + ', CM: ' + str(np.round(conf_matrix_a.reshape(-1), 3))
                    open(file_tmp, 'a').write(s + '\n')

            ####################
            # compute accuracies
            ####################
            acc_dev_dml, acc_dev_bfs, acc_dev_ual, \
            pan_dev_dml, pan_dev_bfs, pan_dev_ual, \
            calibration_dev_dml, calibration_dev_bfs, calibration_dev_ual \
                = self.evaluate_model(docs_L_dev, docs_R_dev, labels_dev, batch_size_dev)

            #####################
            # update progress bar
            #####################
            time = str(datetime.datetime.now()).split('.')[0]
            s = '\n Time: ' + str(time) \
                + '\n Epoch: ' + str(epoch) \
                + '\n ----------' \
                + '\n Acc (dml, train): ' + str(round(100 * (TP_dml_a + TN_dml_a) / (TP_dml_a + FP_dml_a + TN_dml_a + FN_dml_a), 2)) \
                + '\n Acc (bfs, train): ' + str(round(100 * (TP_bfs_a + TN_bfs_a) / (TP_bfs_a + FP_bfs_a + TN_bfs_a + FN_bfs_a), 2)) \
                + '\n Acc (ual, train): ' + str(round(100 * (TP_ual_a + TN_ual_a) / (TP_ual_a + FP_ual_a + TN_ual_a + FN_ual_a), 2)) \
                + '\n ----------' \
                + '\n Acc (dml, dev): ' + str(round(100 * acc_dev_dml, 4)) \
                + '\n Acc (bfs, dev): ' + str(round(100 * acc_dev_bfs, 4)) \
                + '\n Acc (ual, dev): ' + str(round(100 * acc_dev_ual, 4)) \
                + '\n ----------' \
                + '\n PAN scores (dml, dev): ' + str(pan_dev_dml) \
                + '\n PAN scores (bfs, dev): ' + str(pan_dev_bfs) \
                + '\n PAN scores (ual, dev): ' + str(pan_dev_ual) \
                + '\n ----------' \
                + '\n Calibration scores (dml, dev): ' + str(calibration_dev_dml) \
                + '\n Calibration scores (bfs, dev): ' + str(calibration_dev_bfs) \
                + '\n Calibration scores (ual, dev): ' + str(calibration_dev_ual)
            open(file_results, 'a').write(s + '\n')

            ##########################
            # store weights/parameters
            ##########################
            self.store_parameters(epoch, dir_results)
