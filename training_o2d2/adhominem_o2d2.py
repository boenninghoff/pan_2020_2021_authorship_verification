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
from resample_pairs_cal import resample_pairs_cal


class AdHominem_O2D2():
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

    def __init__(self, hyper_parameters, theta_init, theta_E_init):

        # reset graph
        tf.reset_default_graph()

        # hyper-parameters
        self.hyper_parameters = hyper_parameters

        # placeholders for input variables
        self.placeholders, self.thetas_E = self.initialize_placeholders(theta_E_init)

        # batch size
        self.B = tf.shape(self.placeholders['e_w_L'])[0]

        # trainable parameters
        self.theta = self.initialize_parameters(theta_init)
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
            e_d = self.feature_extraction(e_c, e_w, N_w, N_s)

        ############################
        # deep metric learning (DML)
        ############################
        with tf.variable_scope('deep_metric_learning'):
            self.lev_L, self.lev_R = self.metric_layer(e_d=e_d,
                                                       theta=self.theta['metric'],
                                                       dropout=self.dropout['metric'],
                                                       )

        ###################################################
        # kernel distance for probabilstic contrastive loss
        ###################################################
        with tf.variable_scope('Euclidean_distance_and_kernel_function'):
            self.pred_dml = self.compute_kernel_distance(self.lev_L, self.lev_R,
                                                         alpha=self.theta["loss_dml"]["alpha"],
                                                         beta=self.theta["loss_dml"]["beta"],
                                                         )

        ##################################
        # Bayes factor scoring (BFS) layer
        ##################################
        with tf.variable_scope('Bayes_factor_scoring'):
            self.pred_bfs, self.H_W, self.H_B = self.bfs_layer(lev_L=self.lev_L,
                                                               lev_R=self.lev_R,
                                                               theta_bfs_1=self.theta['bfs_1'],
                                                               theta_bfs_2=self.theta['bfs_2'],
                                                               dropout=self.dropout['bfs'],
                                                               )

        ####################################
        # uncertainty adaptation layer (UAL)
        ####################################
        with tf.variable_scope('uncertainty_adaptation_layer'):
            self.pred_ual, self.conf_matrix = self.ual_layer(theta=self.theta['ual'],
                                                             dropout=self.dropout['ual'],
                                                             pred_bfs=self.pred_bfs,
                                                             lev_L=self.lev_L,
                                                             lev_R=self.lev_R,
                                                             )
        #####################################
        # out-of-distribution detector (O2D2)
        #####################################
        with tf.variable_scope('out_of_distribution_detector'):
            self.logits, self.pred_o2d2, self.labels_hat_o2d2 = self.o2d2(lev_L=self.lev_L,
                                                                          lev_R=self.lev_R,
                                                                          theta=self.theta['o2d2'],
                                                                          conf_matrix=self.conf_matrix,
                                                                          )
            self.labels_o2d2 = self.get_o2d2_labels(pred_ual=self.pred_ual[:, 1],
                                                    labels=self.placeholders['labels'],
                                                    thr_0=self.hyper_parameters['thr_0'],
                                                    thr_1=self.hyper_parameters['thr_1'],
                                                    )
        ##############################
        # loss functions and optimizer
        ##############################
        # O2D2 loss
        self.loss = self.loss_o2d2(logits=self.logits, labels=self.labels_o2d2)

        # optimizer
        self.optimizer, self.step = self.prepare_optimizer()

        ################
        # launch session
        ################
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

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

    #####################################
    # Out-Of-Distribution-Detector (O2D2)
    #####################################
    def loss_o2d2(self, logits, labels):

        logits = tf.squeeze(logits)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss)

        return loss

    def o2d2(self, lev_L, lev_R, conf_matrix, theta):

        # input data
        y1 = tf.square(lev_L - lev_R)
        y2 = tf.square(lev_L + lev_R)
        y3 = tf.reshape(conf_matrix, shape=[self.B, 4])
        y = tf.concat([y1, y2, y3], axis=1)

        # layer 1
        is_training = self.placeholders['is_training']
        y = tf.layers.dropout(y, rate=self.hyper_parameters['rate_o2d2'], training=is_training)
        y = tf.nn.xw_plus_b(y, theta["W1"], theta["b1"])
        y = tf.nn.tanh(y)

        # layer 2
        y = tf.layers.dropout(y, rate=self.hyper_parameters['rate_o2d2'], training=is_training)
        y = tf.nn.xw_plus_b(y, theta["W2"], theta["b2"])
        y = tf.nn.tanh(y)

        # layer 3
        y = tf.layers.dropout(y, rate=self.hyper_parameters['rate_o2d2'], training=is_training)
        logits = tf.nn.xw_plus_b(y, theta["W3"], theta["b3"])
        pred = tf.nn.sigmoid(logits)

        y_hat = tf.cast(tf.math.round(tf.squeeze(pred)), dtype=tf.int32)

        return logits, pred, y_hat

    def get_o2d2_labels(self, pred_ual, labels, thr_0=0.45, thr_1=0.55):

        # estimated UAL labels
        labels_hat_ual = tf.cast(tf.math.round(tf.squeeze(pred_ual)), dtype=tf.float32)

        thr_0 = tf.constant(thr_0, dtype=tf.float32)
        thr_1 = tf.constant(thr_1, dtype=tf.float32)

        # make O2D2 labels
        labels_o2d2 = tf.abs(labels_hat_ual - labels)

        condition_0 = tf.where(tf.greater(thr_0, pred_ual),
                               tf.zeros_like(pred_ual, dtype=tf.float32),
                               tf.ones_like(pred_ual, dtype=tf.float32),
                               )
        condition_1 = tf.where(tf.less(thr_1, pred_ual),
                               tf.zeros_like(pred_ual, dtype=tf.float32),
                               tf.ones_like(pred_ual, dtype=tf.float32),
                               )
        condition = tf.multiply(condition_0, condition_1)

        labels_o2d2 = tf.where(tf.cast(condition, dtype=tf.bool),
                               condition,
                               labels_o2d2,
                               )
        return labels_o2d2

    ####################################
    # uncertainty adaptation layer (UAL)
    ####################################
    def ual_layer(self, theta, dropout, pred_bfs, lev_L, lev_R):

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

        return pred_ual, conf_matrix

    ########################################
    # final deep metric learning (DML) layer
    ########################################
    def metric_layer(self, e_d, dropout, theta):

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
        y = tf.nn.xw_plus_b(y, theta['W'], theta['b'])
        # nonlinear output
        y = tf.nn.tanh(y)

        return y[:self.B, :], y[self.B:, :]

    ##################################
    # Bayes factor scoring (BFS) layer
    ##################################
    def bfs_layer(self, lev_L, lev_R, theta_bfs_1, theta_bfs_2, dropout):

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

        return pred, H_W, H_B

    ################################################
    # initialize all placeholders and look-up tables
    ################################################
    def initialize_placeholders(self, theta_E):

        T_c = self.hyper_parameters['T_c']
        T_w = self.hyper_parameters['T_w']
        T_s = self.hyper_parameters['T_s']
        D_c = self.hyper_parameters['D_c']
        D_w = self.hyper_parameters['D_w']

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
            # <PAD> embedding
            E_c_0 = tf.zeros(shape=[1, D_c], dtype=tf.float32)
            # trainable embeddings
            E_c_1 = tf.Variable(theta_E['E_c_1'],
                                name='E_c_1',
                                dtype=tf.float32,
                                trainable=self.hyper_parameters["retrain_chr_emb"],
                                )
            # concatenate zero-padding embedding + trained character embeddings
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
            # <PAD> embedding
            E_w_0 = tf.zeros(shape=[1, D_w], dtype=tf.float32)
            # <UNK> embedding
            E_w_1 = tf.Variable(theta_E['E_w_1'],
                                name='E_w_1',
                                trainable=self.hyper_parameters["retrain_wrd_emb"],
                                dtype=tf.float32,
                                )
            # pre-trained word embedding
            E_w_2 = tf.Variable(theta_E['E_w_2'],
                                name='E_w_2',
                                trainable=self.hyper_parameters["retrain_wrd_emb"],
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
        labels = tf.placeholder(tf.float32, shape=[None], name='labels')
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
                        'labels': labels,
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
        with tf.variable_scope('dropout_metric'):
            dropout['metric'] = {}
            dropout['metric']['metric_1'] = self.make_dropout_mask(shape=[self.B, 2 * D_d],
                                                                   keep_prob=self.hyper_parameters['keep_prob_metric'],
                                                                   )

        with tf.variable_scope('dropout_bfs'):
            dropout['bfs'] = {}
            dropout['bfs']['bfs_1'] = self.make_dropout_mask(shape=[self.B, D_lev],
                                                             keep_prob=self.hyper_parameters['keep_prob_bfs'],
                                                             )
        with tf.variable_scope('dropout_ual'):
            dropout['ual'] = {}
            dropout['ual']['ual_1'] = self.make_dropout_mask(shape=[self.B, D_lev],
                                                             keep_prob=self.hyper_parameters['keep_prob_ual'],
                                                             )
            dropout['ual']['ual_2'] = self.make_dropout_mask(shape=[self.B, 2*D_lev],
                                                             keep_prob=self.hyper_parameters['keep_prob_ual'],
                                                             )
        return dropout

    #######################################
    # initialize trained and new parameters
    #######################################
    def initialize_parameters(self, theta_init):

        D_lev = self.hyper_parameters['D_lev']

        theta = {}

        with tf.variable_scope('theta_cnn'):
            theta['cnn'] = self.initialize_cnn(theta_init['cnn'],
                                               self.hyper_parameters["retrain_cnn"])

        with tf.variable_scope('theta_lstm_ws_forward'):
            theta['lstm_ws_forward'] = self.initialize_lstm(theta_init['lstm_ws_forward'],
                                                            self.hyper_parameters["retrain_bilstm"])
        with tf.variable_scope('theta_lstm_ws_backward'):
            theta['lstm_ws_backward'] = self.initialize_lstm(theta_init['lstm_ws_backward'],
                                                             self.hyper_parameters["retrain_bilstm"])

        with tf.variable_scope('theta_lstm_sd_forward'):
            theta['lstm_sd_forward'] = self.initialize_lstm(theta_init['lstm_sd_forward'],
                                                            self.hyper_parameters["retrain_bilstm"])
        with tf.variable_scope('theta_lstm_sd_backward'):
            theta['lstm_sd_backward'] = self.initialize_lstm(theta_init['lstm_sd_backward'],
                                                             self.hyper_parameters["retrain_bilstm"])

        with tf.variable_scope('theta_att_ws'):
            theta['att_ws'] = self.initialize_att(theta_init['att_ws'],
                                                  self.hyper_parameters["retrain_bilstm"])
        with tf.variable_scope('theta_att_sd'):
            theta['att_sd'] = self.initialize_att(theta_init['att_sd'],
                                                  self.hyper_parameters["retrain_bilstm"])

        with tf.variable_scope('theta_metric'):
            theta['metric'] = self.initialize_mlp(theta_init['metric_a'],
                                                  self.hyper_parameters["retrain_dml"])

        with tf.variable_scope('theta_bfs_1'):
            theta['bfs_1'] = self.initialize_mlp(theta_init['bfs_a_1'],
                                                 self.hyper_parameters["retrain_bfs"])
        with tf.variable_scope('theta_bfs_2'):
            theta['bfs_2'] = self.initialize_bfs(theta_init['bfs_a_2'],
                                                 self.hyper_parameters["retrain_bfs"])

        with tf.variable_scope('theta_ual'):
            theta['ual'] = self.initialize_ual(theta_init['ual_a'],
                                               self.hyper_parameters["retrain_ual"])

        with tf.variable_scope('theta_dml_loss'):
            theta['loss_dml'] = self.initialize_theta_loss(theta_init['loss_dml_a'],
                                                           self.hyper_parameters["retrain_loss_dml"])

        with tf.variable_scope('theta_o2d2'):
            theta['o2d2'] = self.initialize_o2d2(D_lev)

        return theta

    def initialize_o2d2(self, D_lev):

        theta = {"W1": tf.get_variable(name="W1",
                                       shape=[2 * D_lev + 4, 60],
                                       initializer=tf.contrib.layers.xavier_initializer(),
                                       trainable=True,
                                       dtype=tf.float32,
                                       ),
                 "b1": tf.get_variable(name="b1",
                                       shape=[60],
                                       initializer=tf.constant_initializer(0.),
                                       trainable=True,
                                       dtype=tf.float32),
                 "W2": tf.get_variable(name="W2",
                                       shape=[60, 30],
                                       initializer=tf.contrib.layers.xavier_initializer(),
                                       trainable=True,
                                       dtype=tf.float32,
                                       ),
                 "b2": tf.get_variable(name="b2",
                                       shape=[30],
                                       initializer=tf.constant_initializer(0.),
                                       trainable=True,
                                       dtype=tf.float32),
                 "W3": tf.get_variable(name="W3",
                                       shape=[30, 1],
                                       initializer=tf.contrib.layers.xavier_initializer(),
                                       trainable=True,
                                       dtype=tf.float32,
                                       ),
                 "b3": tf.get_variable(name="b3",
                                       shape=[1],
                                       initializer=tf.constant_initializer(0.),
                                       trainable=True,
                                       dtype=tf.float32),
                 }

        return theta

    def initialize_theta_loss(self, theta_init, retrain):
        theta = {'alpha': tf.Variable(theta_init['alpha'],
                                      trainable=retrain,
                                      dtype=tf.float32,
                                      name='loss_alpha',
                                      ),
                 'beta': tf.Variable(theta_init['beta'],
                                     trainable=retrain,
                                     dtype=tf.float32,
                                     name='loss_beta',
                                     ),
                 }

        return theta

    def initialize_ual(self, theta_init, retrain):
        theta = {'W1': tf.Variable(theta_init['W1'],
                                   trainable=retrain,
                                   dtype=tf.float32,
                                   name='W1',
                                   ),
                 'b1': tf.Variable(theta_init['b1'],
                                   trainable=retrain,
                                   dtype=tf.float32,
                                   name='b1',
                                   ),
                 'W2': tf.Variable(theta_init['W2'],
                                   trainable=retrain,
                                   dtype=tf.float32,
                                   name='W2',
                                   ),
                 'b2': tf.Variable(theta_init['b2'],
                                   trainable=retrain,
                                   dtype=tf.float32,
                                   name='b2',
                                   )
                 }

        return theta

    def initialize_bfs(self, theta_init, retrain):
        theta = {'W': tf.Variable(theta_init['W'],
                                  trainable=retrain,
                                  dtype=tf.float32,
                                  name='W',
                                  ),
                 'B': tf.Variable(theta_init['B'],
                                  trainable=retrain,
                                  dtype=tf.float32,
                                  name='B',
                                  ),
                 'mu': tf.Variable(theta_init['mu'],
                                   trainable=retrain,
                                   dtype=tf.float32,
                                   name='mu',
                                   ),
                 }
        return theta

    def initialize_mlp(self, theta_init, retrain):
        theta = {'W': tf.Variable(theta_init['W'],
                                  trainable=retrain,
                                  dtype=tf.float32,
                                  name='W',
                                  ),
                 'b': tf.Variable(theta_init['b'],
                                  trainable=retrain,
                                  dtype=tf.float32,
                                  name='b',
                                  ),
                 }
        return theta

    def initialize_cnn(self, theta_init, retrain):
        theta = {'W': tf.Variable(theta_init['W'],
                                  trainable=retrain,
                                  dtype=tf.float32,
                                  name='W',
                                  ),
                 'b': tf.Variable(theta_init['b'],
                                  trainable=retrain,
                                  dtype=tf.float32,
                                  name='b',
                                  ),
                 }
        return theta

    def initialize_att(self, theta_init, retrain):
        theta = {'W': tf.Variable(theta_init['W'],
                                  trainable=retrain,
                                  dtype=tf.float32,
                                  name='W',
                                  ),
                 'v': tf.Variable(theta_init['v'],
                                  trainable=retrain,
                                  dtype=tf.float32,
                                  name='v',
                                  ),
                 'b': tf.Variable(theta_init['b'],
                                  trainable=retrain,
                                  dtype=tf.float32,
                                  name='b',
                                  ),
                 }
        return theta

    def initialize_lstm(self, theta_init, retrain):
        theta = {'W_i': tf.Variable(theta_init['W_i'],
                                    trainable=retrain,
                                    dtype=tf.float32,
                                    name='W_i',
                                    ),
                 'U_i': tf.Variable(theta_init['U_i'],
                                    trainable=retrain,
                                    dtype=tf.float32,
                                    name='U_i',
                                    ),
                 'b_i': tf.Variable(theta_init['b_i'],
                                    trainable=retrain,
                                    dtype=tf.float32,
                                    name='b_i',
                                    ),
                 'W_f': tf.Variable(theta_init['W_f'],
                                    trainable=retrain,
                                    dtype=tf.float32,
                                    name='W_f',
                                    ),
                 'U_f': tf.Variable(theta_init['U_f'],
                                    trainable=retrain,
                                    dtype=tf.float32,
                                    name='U_f',
                                    ),
                 'b_f': tf.Variable(theta_init['b_f'],
                                    trainable=retrain,
                                    dtype=tf.float32,
                                    name='b_f',
                                    ),
                 'W_c': tf.Variable(theta_init['W_c'],
                                    trainable=retrain,
                                    dtype=tf.float32,
                                    name='W_c',
                                    ),
                 'U_c': tf.Variable(theta_init['U_c'],
                                    trainable=retrain,
                                    dtype=tf.float32,
                                    name='U_c',
                                    ),
                 'b_c': tf.Variable(theta_init['b_c'],
                                    trainable=retrain,
                                    dtype=tf.float32,
                                    name='b_c',
                                    ),
                 'W_o': tf.Variable(theta_init['W_o'],
                                    trainable=retrain,
                                    dtype=tf.float32,
                                    name='W_o',
                                    ),
                 'U_o': tf.Variable(theta_init['U_o'],
                                    trainable=retrain,
                                    dtype=tf.float32,
                                    name='U_o',
                                    ),
                 'b_o': tf.Variable(theta_init['b_o'],
                                    trainable=retrain,
                                    dtype=tf.float32,
                                    name='b_o',
                                    ),
                 }
        return theta

    ######################
    # compute some metrics
    ######################
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

    def compute_labels(self, pred, thr=0.5):
        # numpy array for estimated labels
        labels_hat = np.ones(pred.shape, dtype=np.float32)
        # dissimilar pairs --> 0, similar pairs --> 1
        labels_hat[pred <= thr] = 0
        return labels_hat

    def compute_confidence(self, pred):
        labels_hat = self.compute_labels(pred)
        confidences = pred.copy()
        confidences[labels_hat == 0] = 1.0 - confidences[labels_hat == 0]
        return confidences, labels_hat

    ################
    # get next batch
    ################
    @staticmethod
    def next_batch(t_s, t_e, docs_L, docs_R, labels):
        docs_L_i = docs_L[t_s:t_e]
        docs_R_i = docs_R[t_s:t_e]
        labels_i = labels[t_s:t_e]
        return docs_L_i, docs_R_i, labels_i

    #############################
    # update weights of the model
    #############################
    def update_model(self, x_w_L, x_w_R, x_c_L, x_c_R, labels, N_w_L, N_w_R, N_s_L, N_s_R, lr):

        feed_dict = {self.placeholders['x_w_L']: x_w_L,
                     self.placeholders['x_w_R']: x_w_R,
                     self.placeholders['x_c_L']: x_c_L,
                     self.placeholders['x_c_R']: x_c_R,
                     self.placeholders['labels']: labels,
                     self.placeholders['N_w_L']: N_w_L,
                     self.placeholders['N_w_R']: N_w_R,
                     self.placeholders['N_s_L']: N_s_L,
                     self.placeholders['N_s_R']: N_s_R,
                     self.placeholders['is_training']: True,
                     self.placeholders['lr']: lr,
                     }
        _, loss, pred_dml, pred_bfs, pred_ual, labels_hat_o2d2 \
            = self.sess.run([self.optimizer, self.loss,
                             self.pred_dml, self.pred_bfs, self.pred_ual,
                             self.labels_hat_o2d2,
                             ], feed_dict=feed_dict)

        # define non-responses
        pred_o2d2 = pred_ual[:, 1].copy()
        idx = np.where(labels_hat_o2d2 == 1)[0]
        pred_o2d2[idx] = 0.5
        # number of miss-classification in %
        n_miss = 100 * np.sum(labels_hat_o2d2 == 1) / len(labels)

        return loss, n_miss

    ################
    # evaluate model
    ################
    def evaluate_model(self, docs_L, docs_R, labels, batch_size):

        num_batches = ceil(len(labels) / batch_size)

        pred_dml, pred_bfs, pred_ual, labels_hat_o2d2 = [], [], [], []

        for i in range(num_batches):

            # get next batch
            docs_L_i, docs_R_i, labels_i = self.next_batch(i * batch_size,
                                                           (i + 1) * batch_size,
                                                           docs_L,
                                                           docs_R,
                                                           labels,
                                                           )
            B = len(labels_i)

            if B > 0:
                # word/character embeddings
                x_w_L, N_w_L, N_s_L, x_c_L = self.doc2mat(docs_L_i)
                x_w_R, N_w_R, N_s_R, x_c_R = self.doc2mat(docs_R_i)

                # accuracy for training set
                curr_pred_dml, curr_pred_bfs, curr_pred_ual, curr_labels_o2d2 \
                    = self.compute_eval_measures(x_w_L=x_w_L,
                                                 x_w_R=x_w_R,
                                                 x_c_L=x_c_L,
                                                 x_c_R=x_c_R,
                                                 N_w_L=N_w_L,
                                                 N_w_R=N_w_R,
                                                 N_s_L=N_s_L,
                                                 N_s_R=N_s_R,
                                                 )
                pred_dml.append(curr_pred_dml)
                pred_bfs.append(curr_pred_bfs)
                pred_ual.append(curr_pred_ual)
                labels_hat_o2d2.append(curr_labels_o2d2)

        labels = np.array(labels).reshape(-1)
        labels_hat_o2d2 = np.concatenate(labels_hat_o2d2, axis=0).reshape(-1)

        pred_dml = np.concatenate(pred_dml, axis=0).reshape(-1)
        pred_bfs = np.concatenate(pred_bfs, axis=0).reshape(-1)
        pred_ual = np.concatenate(pred_ual, axis=0).reshape(-1)

        # define non-responses
        pred_o2d2 = pred_ual.copy()
        idx = np.where(labels_hat_o2d2 == 1)[0]
        pred_o2d2[idx] = 0.5

        # number of missclassification in %
        n_miss = 100 * np.sum(labels_hat_o2d2 == 1) / len(labels_hat_o2d2)

        # compute calibration metrics
        confidence_dml, labels_dml = self.compute_confidence(pred_dml)
        calibration_dml = compute_calibration(true_labels=labels,
                                              pred_labels=labels_dml,
                                              confidences=confidence_dml,
                                              )
        confidence_bfs, labels_bfs = self.compute_confidence(pred_bfs)
        calibration_bfs = compute_calibration(true_labels=labels,
                                              pred_labels=labels_bfs,
                                              confidences=confidence_bfs,
                                              )
        confidence_ual, labels_ual = self.compute_confidence(pred_ual)
        calibration_ual = compute_calibration(true_labels=labels,
                                              pred_labels=labels_ual,
                                              confidences=confidence_ual,
                                              )
        confidence_o2d2, labels_o2d2 = self.compute_confidence(pred_o2d2)
        calibration_o2d2 = compute_calibration(true_labels=labels,
                                               pred_labels=labels_o2d2,
                                               confidences=confidence_o2d2,
                                               )
        # compute PAN metrics
        pan_dml = evaluate_all(pred_y=pred_dml, true_y=labels)
        pan_bfs = evaluate_all(pred_y=pred_bfs, true_y=labels)
        pan_ual = evaluate_all(pred_y=pred_ual, true_y=labels)
        pan_o2d2 = evaluate_all(pred_y=pred_o2d2, true_y=labels)

        return pan_dml, pan_bfs, pan_ual, pan_o2d2, \
               calibration_dml, calibration_bfs, calibration_ual, calibration_o2d2, n_miss

    ####################################################
    # evaluate model AdHominem + O2D2 for a single batch
    #####################################################
    def compute_eval_measures(self, x_w_L, x_w_R, x_c_L, x_c_R, N_w_L, N_w_R, N_s_L, N_s_R):

        # compute distances
        pred_dml, pred_bfs, pred_ual, labels_hat_o2d2 = self.sess.run([self.pred_dml, self.pred_bfs, self.pred_ual,
                                                                       self.labels_hat_o2d2],
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

        return pred_dml, pred_bfs, pred_ual[:, 1], labels_hat_o2d2

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
    def doc2mat(self, docs):

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
                                   'weights_o2d2',
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
    def train_model(self, val_set, file_results, file_tmp, dir_results):

        # total number of epochs
        epochs = self.hyper_parameters['epochs']

        # number of batches for dev/test set
        batch_size = self.hyper_parameters['batch_size']
        batch_size_val = self.hyper_parameters['batch_size_val']

        # dev set
        docs_L_val, docs_R_val, labels_val = val_set

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

            # load calibration (training) set
            docs_L_tr, docs_R_tr, labels_tr = resample_pairs_cal(file_results)
            # shuffle (again)
            docs_L_tr, docs_R_tr, labels_tr = shuffle(docs_L_tr, docs_R_tr, labels_tr)

            # number of training pairs
            N_tr = len(labels_tr)
            # number of batches for training
            num_batches_tr = ceil(N_tr / batch_size)

            # average loss and accuracy
            loss = []

            # loop over all batches
            for i in range(num_batches_tr):

                # get next batch
                docs_L_i, docs_R_i, labels_i = self.next_batch(i * batch_size,
                                                               (i + 1) * batch_size,
                                                               docs_L_tr,
                                                               docs_R_tr,
                                                               labels_tr,
                                                               )

                # current batch size
                B = len(labels_i)

                if B > 0:
                    # word / character embeddings
                    x_w_L, N_w_L, N_s_L, x_c_L = self.doc2mat(docs_L_i)
                    x_w_R, N_w_R, N_s_R, x_c_R = self.doc2mat(docs_R_i)

                    # update model parameters
                    curr_loss, n_miss \
                        = self.update_model(x_w_L=x_w_L,
                                            x_w_R=x_w_R,
                                            x_c_L=x_c_L,
                                            x_c_R=x_c_R,
                                            labels=np.array(labels_i),
                                            N_w_L=N_w_L,
                                            N_w_R=N_w_R,
                                            N_s_L=N_s_L,
                                            N_s_R=N_s_R,
                                            lr=lr[epoch],
                                            )

                    # add to lists
                    loss.append(curr_loss)

                    s = 'Epoch:' + str(epoch) \
                        + ', BatchSize: ' + str(round(100 * (i + 1) / num_batches_tr, 0)) \
                        + ', #Pairs: ' + str(N_tr) \
                        + ', Loss: ' + str(round(float(np.mean(loss)), 3)) \
                        + ', LernRate: ' + str(round(lr[epoch], 5)) \
                        + ', #NonResponse: ' + str(np.round(n_miss, 3))
                    open(file_tmp, 'a').write(s + '\n')

            ####################
            # compute accuracies
            ####################
            open(file_tmp, 'a').write('Epoch training done, start inference...')
            pan_val_dml, pan_val_bfs, pan_val_ual, pan_val_o2d2, \
            calibration_val_dml, calibration_val_bfs, calibration_val_ual, calibration_val_o2d2, n_miss \
                = self.evaluate_model(docs_L_val, docs_R_val, labels_val, batch_size_val)

            #####################
            # update progress bar
            #####################
            time = str(datetime.datetime.now()).split('.')[0]
            s = '\n Time: ' + str(time) \
                + '\n Epoch: ' + str(epoch) \
                + '\n ----------' \
                + '\n PAN scores (dml, val): ' + str(pan_val_dml) \
                + '\n PAN scores (bfs, val): ' + str(pan_val_bfs) \
                + '\n PAN scores (ual, val): ' + str(pan_val_ual) \
                + '\n PAN scores (o2d2, val): ' + str(pan_val_o2d2) \
                + '\n ----------' \
                + '\n Calibration scores (dml, val): ' + str(calibration_val_dml) \
                + '\n Calibration scores (bfs, val): ' + str(calibration_val_bfs) \
                + '\n Calibration scores (ual, val): ' + str(calibration_val_ual) \
                + '\n Calibration scores (o2d2, val): ' + str(calibration_val_o2d2) \
                + '\n ----------' \
                + '\n # Non-responses: ' + str(np.round(n_miss, 3))
            open(file_results, 'a').write(s + '\n')

            ##########################
            # store weights/parameters
            ##########################
            self.store_parameters(epoch, dir_results)
