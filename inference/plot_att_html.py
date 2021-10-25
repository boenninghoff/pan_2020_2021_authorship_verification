import re
import numpy as np
import os
import pickle


def sliding_window(doc, T_w, hop_length):
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


###################################
# function to create html documents
###################################
def visualize(doc_L, alpha_s_L, alpha_w_L,
              doc_R, alpha_s_R, alpha_w_R,
              T_s, T_w, n, d, l, dir_html,
              weighting=True, rescaling=True,
              ):

    def func_cleaning(sent):
        """
            remove "<" and ">": <UNK>  --> UNK
        """
        sent = ' '.join(re.sub('<', '', sent).split())
        sent = ' '.join(re.sub('>', '', sent).split())
        return sent

    def func_weighting(alpha_s, alpha_w):
        """
            alpha_w[t, w] = alpha_w[t, w] * alpha_s[t]
        """
        T_s, T_w = alpha_w.shape

        for i in range(T_s):
            # weighting: alpha_s[i] * alpha_w[i, j] for all j
            alpha_w[i, :] = alpha_s[i] * alpha_w[i, :]

        return alpha_w

    def func_rescaling(alpha_s, alpha_w):
        """
            min-max normalization to obtain values between 0 and 1

        """
        # rescale alpha_s between [0, 1]
        alpha_s = np.array(alpha_s).reshape(-1)
        max_alpha = np.max(alpha_s)
        min_alpha = np.min(alpha_s)
        alpha_s = (alpha_s - min_alpha) / (max_alpha - min_alpha)

        # rescale alpha_w between [0, 1]
        alpha_w = np.array(alpha_w).reshape(-1)
        max_alpha = np.max(alpha_w)
        min_alpha = np.min(alpha_w)
        alpha_w = (alpha_w - min_alpha) / (max_alpha - min_alpha)

        return alpha_s.reshape((T_s, 1)), alpha_w.reshape((T_s, T_w))

    #################################################
    # weighting: alpha_s[i] * alpha_w[i, j] for all j
    #################################################
    if weighting:
        alpha_w_L = func_weighting(alpha_s_L, alpha_w_L)
        alpha_w_R = func_weighting(alpha_s_R, alpha_w_R)

    ################################################
    # rescaling: (alpha_w[i, j] - min) / (max - min)
    ################################################
    if rescaling:
        alpha_s_L, alpha_w_L = func_rescaling(alpha_s_L, alpha_w_L)
        alpha_s_R, alpha_w_R = func_rescaling(alpha_s_R, alpha_w_R)

    with open(os.path.join(dir_html, str(n) + ".html"), "w") as html_file:

        whitespace_token = "  "

        #############
        # first lines
        #############
        html_file.write("<!DOCTYPE html>" + "\n" + "<html>" + "\n" + "<body>")
        html_file.write("<h1 style=\"font-size:12px\">")
        html_file.write('<b> prediction = %.2f, label = %s </b> <br> <br>' % (d, l))
        html_file.write("</h1>")

        ################
        #  first document
        ################
        html_file.write("<p style=\"font-size:12px\">" + "<b> document 1 </b> <br> <br>")
        for i, sentence in enumerate(doc_L[:T_s]):
            # <UNK> --> UNK
            sent = func_cleaning(sentence)
            # display sentence-based attentions
            html_file.write(' <font style="background: rgba(0, 0, 255, %f)">&nbsp&nbsp&nbsp&nbsp&nbsp</font>' % (alpha_s_L[i, 0]))
            tokens = sent.split()[:T_w]
            # display word-based attentions
            html_file.write('<font style="background: rgba(0, 0, 0, %f)">%s</font>' % (0.0, whitespace_token))
            for j, token in enumerate(tokens):
                html_file.write('<font style="background: rgba(255, 0, 0, %f)">%s</font>' % (alpha_w_L[i, j], token))
                html_file.write('<font style="background: rgba(0, 0, 0, %f)">%s</font>' % (0.0, whitespace_token))
            html_file.write('<br>')

        html_file.write("</p>")
        html_file.write('<br>')

        ##################
        #  second document
        ##################
        html_file.write("<p style=\"font-size:12px\">" + "<b> document 2 </b> <br> <br>")
        for i, sentence in enumerate(doc_R[:T_s]):
            # <UNK> --> UNK
            sent = func_cleaning(sentence)
            # display sentence-based attentions
            html_file.write('<font style="background: rgba(0, 0, 255, %f)">&nbsp&nbsp&nbsp&nbsp&nbsp</font>' % (alpha_s_R[i, 0]))
            tokens = sent.split()[:T_w]
            # display word-based attentions
            html_file.write('<font style="background: rgba(0, 0, 0, %f)">%s</font>' % (0.0, whitespace_token))
            for j, token in enumerate(tokens):
                html_file.write('<font style="background: rgba(255, 0, 0, %f)">%s</font>' % (alpha_w_R[i, j], token))
                html_file.write('<font style="background: rgba(0, 0, 0, %f)">%s</font>' % (0.0, whitespace_token))
            html_file.write('<br>')
        html_file.write("</p>")

        #################
        # end of document
        #################
        html_file.write("</body>" + "\n" + "</html>")


# chosen document
i = 10

dir_data = os.path.join("..", "data_preprocessed")
dir_results = os.path.join("..", "results_o2d2")

# create folder for plots
if not os.path.exists(os.path.join(dir_results, 'plots')):
    os.makedirs(os.path.join(dir_results, "plots"))
dir_html = os.path.join(dir_results, 'plots', 'htmls')
if not os.path.exists(dir_html):
    os.makedirs(dir_html)

path = os.path.join(dir_results, "weights_o2d2", "weights_0")
with open(path, 'rb') as f:
    parameters = pickle.load(f)
T_w = parameters['hyper_parameters']["T_w"]
T_s = parameters['hyper_parameters']["T_s"]
hop_length = parameters['hyper_parameters']["hop_length"]

# load attentions and predictions
with open(os.path.join(dir_results, "results_att_lev_pred"), 'rb') as f:
    _, _, _, pred_o2d2, _, _, _, _, _, _, _, _, _, _, _, _, att_w_L, att_w_R, att_s_L, att_s_R = pickle.load(f)


# load docs and groundtruth labels
with open(os.path.join(dir_data, "pairs_val"), 'rb') as f:
    docs_L, docs_R, labels, _ = pickle.load(f)
labels = np.array(labels)

doc_L = sliding_window(docs_L[i], T_w, hop_length)
doc_R = sliding_window(docs_R[i], T_w, hop_length)

#####################
# make html documents
######################
visualize(doc_L, att_s_L[i, :], att_w_L[i, :, :],
          doc_R, att_s_R[i, :], att_w_R[i, :, :],
          T_s, T_w, i, pred_o2d2[i], labels[i],
          dir_html,
          )

