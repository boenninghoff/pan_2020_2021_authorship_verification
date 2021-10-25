import os
import pickle
import numpy as np
from matplotlib import pyplot as plt


def sort_labels(predictions, confidences, labels_a, labels_c, labels_hat):

    pred_SA_DF = []
    pred_DA_SF = []

    conf_SA_DF = []
    conf_DA_SF = []

    labels_SA_DF = []
    labels_DA_SF = []

    labels_hat_SA_DF = []
    labels_hat_DA_SF = []

    for i in range(len(confidences)):
        if labels_a[i] == 1 and labels_c[i] == 0:
            pred_SA_DF.append(predictions[i])
            conf_SA_DF.append(confidences[i])
            labels_SA_DF.append(labels_a[i])
            labels_hat_SA_DF.append(labels_hat[i])
        elif labels_a[i] == 0 and labels_c[i] == 1:
            pred_DA_SF.append(predictions[i])
            conf_DA_SF.append(confidences[i])
            labels_DA_SF.append(labels_a[i])
            labels_hat_DA_SF.append(labels_hat[i])
        
    return np.array(pred_SA_DF), np.array(pred_DA_SF), \
           np.array(conf_SA_DF), np.array(conf_DA_SF), \
           np.array(labels_hat_SA_DF), np.array(labels_hat_DA_SF), \
           np.array(labels_SA_DF), np.array(labels_DA_SF)


def binning(confidences, labels_hat, labels, num_bins):

    I = [m / num_bins for m in range(num_bins + 1)]
    B = []
    for i in range(num_bins):
        B_m = [(p, label_hat, label_true) for p, label_hat, label_true in zip(confidences, labels_hat, labels)
               if I[i] < p + 1e-06 <= I[i+1]]
        B.append(B_m)
    return B


def acc(B):
    acc = []
    for B_m in B:
        acc.append(acc_m(B_m))
    return acc


def acc_m(B_m):
    hit_count = 0
    for sample in B_m:
        if sample[1] == sample[2]:
            hit_count += 1
    if len(B_m) == 0:
        return -1
    else:
        return hit_count / len(B_m)


def count(B):
    count = []
    for B_m in B:
        count.append(count_m(B_m))
    return count


def count_m(B_m):
    count_collected = 0
    for _ in B_m:
        count_collected += 1
    return count_collected


def get_conf(B):
    conf = []
    for B_m in B:
        conf.append(conf_m(B_m))
    return conf


def conf_m(B_m):
    p_collected = 0
    for sample in B_m:
        p_collected += sample[0]
    if len(B_m) == 0:
        return -1
    else:
        return p_collected / len(B_m)