"""
This is for evaluation metrics calculation during validation and testing phases.

Author: Tao Qiqi
Date: 2024-03
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

# class FairnessEvaluator(object):
#     def __init__(self, labels, preds, intersec_labels):
#         self.auc = roc_auc_score(labels, preds)
#         self.acc = sum(labels == preds) / len(labels)
#         self.con_mat = confusion_matrix(labels, preds)
#
#     def auc(self):
def detection_metrics(labels, pred_labels, pred_probs):
    """
    Calculate detection accuracy, AUC, True Positive Rate, False Positive Rate and confusion matrix.
    :param labels: np.array
    :param pred_labels: np.array
    :param pred_probs: np.array
    :return:
    """
    con_mat = confusion_matrix(labels, pred_labels)
    if con_mat.shape[0] != 2:
        TN = con_mat
        con_mat = np.zeros((2,2))
        con_mat[0,0] = TN
    TN = con_mat[0][0]
    FN = con_mat[1][0]
    TP = con_mat[1][1]
    FP = con_mat[0][1]
    acc = (TN+TP) / (TN+FN+TP+FP)
    if len(set(labels)) > 1:
        auc = roc_auc_score(labels, pred_probs)
        FPR = FP / (FP + TN)
        TPR = TP/(TP+FN)
    else:
        auc = 0
        TPR = 0
        FPR = 0
    return acc, auc, TPR, FPR, con_mat

# fairness metrics: F_FPR, F_OAE, F_MEO
def fairness_metrics(labels, pred_labels, pred_probs, demographic_groups):
    """

    :param labels: np.array
    :param pred_labels: np.array
    :param pred_probs: np.array
    :param demographic_groups: list
    :return:
    """
    metrics = {}
    overall_acc, overall_auc, overall_TPR, overall_FPR, overall_con_mat = detection_metrics(labels,pred_labels,pred_probs)
    f_fpr = 0
    for _, group in enumerate(set(demographic_groups)):
        indices = np.array([x==group for x in demographic_groups])
        group_labels = labels[indices]
        group_pred_labels = pred_labels[indices]
        group_pred_probs = pred_probs[indices]
        g_acc, g_auc, g_TPR, g_FPR, g_con_mat = detection_metrics(group_labels, group_pred_labels, group_pred_probs)
        metrics[group] = {'group_acc': g_acc,
                          'group_auc': g_auc,
                          'group_TPR': g_TPR,
                          'group_FPR': g_FPR,
                          'group_confusion_matrix': g_con_mat}
        f_fpr += abs(g_FPR - overall_FPR)

    metrics_df = pd.DataFrame(metrics).T
    ## calculate F_OAE
    f_oae = max(metrics_df['group_acc'])

    ## calculate F_MEO
    component1 = max(metrics_df['group_TPR']) - min(metrics_df['group_TPR'])
    component2 = max(metrics_df['group_FPR']) - min(metrics_df['group_FPR'])
    component3 = max(1-metrics_df['group_TPR']) - min(1-metrics_df['group_TPR'])
    component4 = max(1-metrics_df['group_FPR']) - min(1-metrics_df['group_FPR'])
    f_meo = max(component1, component2, component3, component4)

    return overall_acc, overall_auc, overall_FPR, f_fpr, f_oae, f_meo, metrics

if __name__ == "__main__":
    labels = np.array([0,1,0,0,0])
    preds = np.array([0,1,0,1,1])
    x = labels == preds

    demographic_groups = [1,1,2,1,2]
    f_fpr, f_oae, f_meo, fair_metrics = fairness_metrics(labels,preds,preds,demographic_groups)
    print(f'F_FPR: {f_fpr}.')
    print(f'F_OAE: {f_oae}.')
    print(f'F_MEO: {f_meo}.')
    print(fair_metrics)