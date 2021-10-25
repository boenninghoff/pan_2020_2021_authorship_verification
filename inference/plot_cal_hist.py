# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.join('..', 'helper_functions'))
from utils import sort_labels
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import os
import pickle

##############
# prepare data
##############
dir_data = os.path.join("..", "data_preprocessed")
dir_results = os.path.join("..", "results_o2d2")
if not os.path.exists(os.path.join(dir_results, 'plots')):
    os.makedirs(os.path.join(dir_results, "plots"))

with open(os.path.join(dir_results, "results_att_lev_pred"), 'rb') as f:
    pred_dml, pred_bfs, pred_ual, pred_o2d2, _, _, _, _, conf_dml, conf_bfs, conf_ual, conf_o2d2, labels_dml, labels_bfs, labels_ual, labels_o2d2, _, _, _, _ = pickle.load(f)

# load docs and groundtruth labels
with open(os.path.join(dir_data, "pairs_val"), 'rb') as f:
    _, _, labels_a, labels_c = pickle.load(f)
labels_a = np.array(labels_a)
labels_c = np.array(labels_c)

#############
# sort labels
#############
# DML
pred_SA_DF_dml, pred_DA_SF_dml, \
conf_SA_DF_dml, conf_DA_SF_dml, \
labels_hat_SA_DF_dml, labels_hat_DA_SF_dml, \
labels_SA_DF_dml, labels_DA_SF_dml \
    = sort_labels(pred_dml, conf_dml, labels_a, labels_c, labels_dml)
# BFS
pred_SA_DF_bfs, pred_DA_SF_bfs, \
conf_SA_DF_bfs, conf_DA_SF_bfs, \
labels_hat_SA_DF_bfs, labels_hat_DA_SF_bfs, \
labels_SA_DF_bfs, labels_DA_SF_bfs \
    = sort_labels(pred_bfs, conf_bfs, labels_a, labels_c, labels_bfs)
# UAL
pred_SA_DF_ual, pred_DA_SF_ual, \
conf_SA_DF_ual, conf_DA_SF_ual, \
labels_hat_SA_DF_ual, labels_hat_DA_SF_ual, \
labels_SA_DF_ual, labels_DA_SF_ual \
    = sort_labels(pred_ual, conf_ual, labels_a, labels_c, labels_ual)

# accuracies
accuracy_SA_DF_dml = accuracy_score(labels_hat_SA_DF_dml, labels_SA_DF_dml)
accuracy_DA_SF_dml = accuracy_score(labels_hat_DA_SF_dml, labels_DA_SF_dml)
accuracy_SA_DF_bfs = accuracy_score(labels_hat_SA_DF_bfs, labels_SA_DF_bfs)
accuracy_DA_SF_bfs = accuracy_score(labels_hat_DA_SF_bfs, labels_DA_SF_bfs)
accuracy_SA_DF_ual = accuracy_score(labels_hat_SA_DF_ual, labels_SA_DF_ual)
accuracy_DA_SF_ual = accuracy_score(labels_hat_DA_SF_ual, labels_DA_SF_ual)

##########
# plotting
##########
binwidth = 1.0 / 10

alpha_w = 0.6
alpha_c = 0.6


def plot_histogram(pred, acc, conf, color, ax, type):

    sns.histplot(pred,
                 stat="probability",
                 legend=True,
                 label=type,
                 alpha=alpha_w,
                 color=color,
                 binwidth=binwidth,
                 binrange=(0, 1),
                 )
    ax.vlines(acc,
               0, 1,
               colors='black',
               linestyles='solid',
               label=f'Accuracy: {round(acc * 100, 2)}%',
               )
    ax.vlines(np.mean(conf), 0, 1,
               colors='black',
               linestyles='dotted',
               label=f'Avg. confidence: {round(np.mean(conf) * 100, 2)}%',
               )
    ax.legend()  # loc='upper left')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("% of samples")
    plt.xlabel("Output predictions")
    plt.grid()


fig = plt.figure(figsize=[20, 12])

# DML
ax1 = fig.add_subplot(2, 3, 1)
plot_histogram(pred_SA_DF_dml, accuracy_SA_DF_dml, conf_SA_DF_dml, "orange", ax1, type="SA_DF (DML)")
ax2 = fig.add_subplot(2, 3, 4)
plot_histogram(pred_DA_SF_dml, accuracy_DA_SF_dml, conf_DA_SF_dml, "blue", ax2, type="DA_SF (DML)")

# BFS
ax1 = fig.add_subplot(2, 3, 2)
plot_histogram(pred_SA_DF_bfs, accuracy_SA_DF_bfs, conf_SA_DF_bfs, "orange", ax1, type="SA_DF (BFS)")
ax2 = fig.add_subplot(2, 3, 5)
plot_histogram(pred_DA_SF_bfs, accuracy_DA_SF_bfs, conf_DA_SF_bfs, "blue", ax2, type="DA_SF (BFS)")

# UAL
ax1 = fig.add_subplot(2, 3, 3)
plot_histogram(pred_SA_DF_ual, accuracy_SA_DF_ual, conf_SA_DF_ual, "orange", ax1, type="SA_DF (UAL)")
ax2 = fig.add_subplot(2, 3, 6)
plot_histogram(pred_DA_SF_ual, accuracy_DA_SF_ual, conf_DA_SF_ual, "blue", ax2, type="DA_SF (UAL)")

plt.savefig(os.path.join(dir_results, 'plots', 'cal_histogram.png'), bbox_inches='tight')

