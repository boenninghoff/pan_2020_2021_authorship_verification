# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.join('..', 'helper_functions'))
from utils import binning, acc, get_conf, count
import numpy as np
from matplotlib import pyplot as plt
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
    _, _, labels, _ = pickle.load(f)
labels = np.array(labels)


def plot_reliability_diagram(pred, conf, labels, labels_hat, layer, ax, num_bins=10):
    
    color = "firebrick"
    
    if conf is not None:
    
        B = binning(conf, labels_hat, labels, num_bins)
        acc_ = acc(B)
        conf_ = get_conf(B)

        bin_size = 1 / num_bins
        positions = []
        akk_pos = bin_size / 2
        for i in range(num_bins):
            if acc_[i] == -1:
                positions.append(0)
            else:
                positions.append(akk_pos)
            akk_pos += bin_size
    
        acc_step = [x + bin_size / 2 for x in acc_]
    
        acc_step_norm = []
        for data in acc_step:
            if data < 1:
                acc_step_norm.append(data)
            else:
                acc_step_norm.append(1)
        acc_step = acc_
        tilde = 0.01
        gap_color = color
        counter = count(B)
        for i in range(num_bins):

            ###########################################################
            # define gap_alpha to visualize number of samples per bin
            ###########################################################
            x = counter[i] / (np.max(counter))
            gap_alpha = x + (1 / np.exp(15.42 * x + 0.693147)) - 0.000001
    
            ###########################################################
            # draw reliability diagram
            # white bars to only highlight the gap
            ###########################################################
            if conf_[i] > acc_step[i]:
                if abs(acc_step[i] - conf_[i]) > tilde:
                    gap = ax.bar(positions[i],
                                 positions[i],
                                 width=bin_size,
                                 alpha=gap_alpha,
                                 color=gap_color,
                                 label="Gap",
                                 )
                black = ax.bar(positions[i],
                               acc_step[i],
                               width=bin_size,
                               alpha=1,
                               linewidth=0,
                               color="black",
                               label="Accuracy",
                               )
                white = ax.bar(positions[i],
                               acc_step[i] - 0.007,
                               width=bin_size + 0.003,
                               alpha=1,
                               linewidth=0.1,
                               color="white",
                               )
            else:
                black = ax.bar(positions[i],
                               acc_step[i],
                               width=bin_size,
                               alpha=1,
                               linewidth=0,
                               color="black",
                               label="Accuracy",
                               )
                for_line = ax.bar(positions[i],
                                  acc_step[i] - 0.007,
                                  width=bin_size,
                                  alpha=1,
                                  linewidth=0.1,
                                  color="white",
                                  )
                if abs(acc_step[i] - conf_[i]) > tilde:
                    gap = ax.bar(positions[i],
                                 acc_step[i] - 0.007,
                                 width=bin_size,
                                 alpha=gap_alpha,
                                 linewidth=0,
                                 color=gap_color,
                                 label="Gap",
                                 )
                optimal = ax.bar(positions[i],
                                 positions[i],
                                 width=bin_size,
                                 alpha=1,
                                 color="white",
                                 edgecolor="white",
                                 linewidth=0.1,
                                 )
    
        acc_label = ax.bar(-1, -1, color="black", label="Accuracy")
        gap_label = ax.bar(-1, -1, color=gap_color, label="Gap")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.plot([0, 1], [0, 1], linestyle=":", color="gray")
        plt.xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.xlabel("Confidence (" + layer + ")")
        plt.ylabel("Accuracy")
        plt.xlim([0.5, 1])
        plt.ylim([0.5, 1])
        plt.legend(handles=[acc_label, gap_label], loc='lower right')


num_bins = 10
fig = plt.figure(figsize=[15, 15])
ax1 = fig.add_subplot(1, 3, 1)
plot_reliability_diagram(pred_dml, conf_dml, labels, labels_dml, "DML", ax=ax1, num_bins=num_bins)
ax2 = fig.add_subplot(1, 3, 2)
plot_reliability_diagram(pred_bfs, conf_bfs, labels, labels_bfs, "BFS", ax=ax2, num_bins=num_bins)
ax3 = fig.add_subplot(1, 3, 3)
plot_reliability_diagram(pred_ual, conf_ual, labels, labels_ual, "UAL", ax=ax3, num_bins=num_bins)
plt.savefig(os.path.join(dir_results, 'plots', 'rel_diagram.png'), bbox_inches='tight')
