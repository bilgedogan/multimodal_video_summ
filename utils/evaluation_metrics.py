# -*- coding: utf-8 -*-
import torch
from torchmetrics.regression import SpearmanCorrCoef, KendallRankCorrCoef
import pdb
import csv
from collections import Counter
from scipy import stats
import numpy as np
import torch

# 텐서 출력 옵션 변경: 생략 없이 전체 출력
torch.set_printoptions(threshold=torch.inf)

def evaluate_summary_f1(predicted_summary, user_summary, eval_metric='avg'):
    """ Compute keyshot F1-score between a binary predicted summary and one or more
    binary user summaries.

    :param Tensor predicted_summary: Binary per-frame vector (n_frames,).
    :param Tensor user_summary: Binary per-frame vectors (n_users, n_frames).
    :param str eval_metric: 'max' (SumMe convention) or 'avg' (TVSum convention).
    """
    predicted_summary = np.array(predicted_summary.cpu())
    user_summary = np.array(user_summary.cpu())

    n_users = user_summary.shape[0]
    f_scores = []
    for u in range(n_users):
        gt = user_summary[u]
        overlap = np.sum(predicted_summary * gt)
        precision = overlap / (np.sum(predicted_summary) + 1e-8)
        recall = overlap / (np.sum(gt) + 1e-8)
        f_scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))

    return max(f_scores) if eval_metric == 'max' else float(np.mean(f_scores))

def evaluate_summary(predicted_summary, user_summary, video_name, pred_score, eval_data):
    """ Compare the predicted summary with the user defined one(s).

    :param ndarray predicted_summary: The generated summary from our model.
    :param ndarray gt_summary: The user defined ground truth summaries (or summary).
    """
    y_pred2=predicted_summary
    y_true2=user_summary.mean(axis=0)

    if eval_data == 'summe':
        curr_rho_coeff = stats.spearmanr(y_pred2.cpu(), y_true2.cpu())[0]
        curr_tau_coeff = stats.kendalltau(stats.rankdata(-np.array(y_pred2.cpu())), stats.rankdata(-np.array(y_true2.cpu())))[0]
        f1 = evaluate_summary_f1(predicted_summary, user_summary, eval_metric='max')
        return curr_tau_coeff, curr_rho_coeff, f1

    else:
        with open('TVSum/ydata-tvsum50-anno.tsv') as annot_file:
            annot = list(csv.reader(annot_file, delimiter="\t"))
        annotation_length = list(Counter(np.array(annot)[:, 0]).values())
        user_scores = []
        for idx in range(1,51):
            init = (idx - 1) * annotation_length[idx-1]
            till = idx * annotation_length[idx-1]
            user_score = []
            for row in annot[init:till]:
                curr_user_score = row[2].split(",")
                curr_user_score = np.array([float(num) for num in curr_user_score])
                curr_user_score = curr_user_score / curr_user_score.max(initial=-1)
                curr_user_score = curr_user_score[::15]

                user_score.append(curr_user_score)
            user_scores.append(user_score)

        user = int(video_name.split("_")[-1])

        curr_user_score = user_scores[user-1]

        tmp_rho_coeff = []
        tmp_tau_coeff = []
        for annotation in range(len(curr_user_score)):
            true_user_score = curr_user_score[annotation]

            curr_rho_coeff = stats.spearmanr(pred_score.cpu(), true_user_score)[0]
            curr_tau_coeff = stats.kendalltau(stats.rankdata(-np.array(pred_score.cpu())), stats.rankdata(-np.array(true_user_score)))[0]

            tmp_rho_coeff.append(curr_rho_coeff)
            tmp_tau_coeff.append(curr_tau_coeff)
        pS = np.mean(tmp_rho_coeff)
        kT = np.mean(tmp_tau_coeff)
        f1 = evaluate_summary_f1(predicted_summary, user_summary, eval_metric='avg')
        return kT, pS, f1