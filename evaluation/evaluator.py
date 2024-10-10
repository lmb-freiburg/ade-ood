import numpy as np
import torch
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, roc_auc_score, average_precision_score, f1_score
from datasets.ade_ood import ADEOoDDataset
import os


def fpr_at_tpr(scores, ood_gt, tpr=0.95):
    fprs, tprs, _ = roc_curve(ood_gt, scores)
    idx = np.argmin(np.abs(np.array(tprs) - tpr))
    return fprs[idx]

def f_max_score(scores, ood_gt):
    precision, recall, thresholds = precision_recall_curve(ood_gt, scores)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-6)
    f_max = np.max(f1_scores)
    return f_max

def ap(scores, ood_gt):
    return average_precision_score(ood_gt, scores)

def f1(scores, ood_gt):
    return f1_score(ood_gt, scores)

def auroc(scores, target):
    return roc_auc_score(target, scores)


class StreamingEval:
    def __init__(self, ood_id, ignore_ids=255):
        self.ood_id = ood_id
        self.collected_scores = []
        self.collected_gts = []
        if isinstance(ignore_ids, int):
            ignore_ids = [ignore_ids]
        self.ignore_ids = ignore_ids

    def add(self, scores, segm_gt):
        valid = np.logical_not(np.in1d(segm_gt, self.ignore_ids))
        ood_gt = (segm_gt == self.ood_id)
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores)
        if not isinstance(ood_gt, torch.Tensor):
            ood_gt = torch.tensor(ood_gt)
        ood_gt = ood_gt.cpu().flatten()[valid]
        self.collected_scores.append(scores.cpu().flatten()[valid])
        self.collected_gts.append(ood_gt)

    def get_scores_and_labels(self):
        all_scores = torch.cat(self.collected_scores, 0)
        all_gts = torch.cat(self.collected_gts, 0)
        return all_scores, all_gts

    def get_ap(self):
        scores, labels = self.get_scores_and_labels()
        return ap(scores, labels.int()) * 100
    
    def get_auroc(self):
        scores, labels = self.get_scores_and_labels()
        return auroc(scores, labels.int()) * 100
    
    def get_fpr95(self):
        scores, labels = self.get_scores_and_labels()
        return fpr_at_tpr(scores, labels.int()) * 100

    def get_fmax(self):
        scores, labels = self.get_scores_and_labels()
        return f_max_score(scores, labels.int()) * 100

    def get_pr_curve(self):
        scores, labels = self.get_scores_and_labels()
        return precision_recall_curve(labels, scores)


def ade_ood_eval_with_callback(method_callback, data_preprocess_callback=None, ade_ood_path=None):
    """
    Evaluate the method using a callback function.
    :param method_callback: Your method's pipeline. It should take a single input argument (the test image) and return the OoD score map as a 2D torch/numpy array.
    :param data_preprocess_callback: A function that preprocesses the test images before passing it to the method_callback.
    :param ade_ood_path: Path to the ADE OoD dataset. If not provided, it will be read from the $ADE_OOD_PATH environment variable.

    :return: AP and FPR at 95% TPR.
    """
    ade_ood = ADEOoDDataset(ade_ood_path or os.path.expandvars('$ADE_OOD_PATH'))
    evaluator = StreamingEval(ade_ood.ood_idx)

    for img, gt, _ in ade_ood:
        if data_preprocess_callback is not None:
            img = data_preprocess_callback(img)
        score_map = method_callback(img)
        evaluator.add(score_map, gt)

    ap, fpr_at_tpr = evaluator.get_ap(), evaluator.get_fpr95()
    return ap, fpr_at_tpr


def ade_ood_eval_with_scores_from_disk(scores_path, ade_ood_path=None, scores_suffix='_dood_scores.npy'):
    """
    Evaluate the method using scores saved on disk.
    :param scores_path: Path to the directory containing the scores. The scores should be saved as .npy files with the same name as the images.
    :param ade_ood_path: Path to the ADE OoD dataset. If not provided, it will be read from the $ADE_OOD_PATH environment variable.

    :return: AP and FPR at 95% TPR.
    """
    ade_ood = ADEOoDDataset(ade_ood_path or os.path.expandvars('$ADE_OOD_PATH'))
    evaluator = StreamingEval(ade_ood.ood_idx)

    for img, gt, img_name in ade_ood:
        scores = np.load(os.path.join(scores_path, os.path.splitext(img_name)[0] + scores_suffix))
        evaluator.add(scores, gt)

    ap, fpr_at_tpr = evaluator.get_ap(), evaluator.get_fpr95()
    return ap, fpr_at_tpr