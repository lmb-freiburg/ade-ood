import os
from . import ADEOoDDataset, StreamingEval, ade_ood_eval_with_callback, ade_ood_eval_with_scores_from_disk

dummy_method = lambda x: x.mean(-1)

if __name__ == '__main__':
    # Example 1: explicit usage of the dataset and evaluator
    ade_ood = ADEOoDDataset(os.path.expandvars('$ADE_OOD_PATH'))
    evaluator = StreamingEval(ade_ood.ood_idx)
    for img, gt, img_name in ade_ood:
        score_map = dummy_method(img)
        evaluator.add(score_map, gt)
    ap, fpr95 = evaluator.get_ap(), evaluator.get_fpr95()
    print(f'AP: {ap:.2f}, FPR95: {fpr95:.2f}')

    # Example 2: passing your method as a callback (see method signature for options)
    ap, fpr95 = ade_ood_eval_with_callback(dummy_method)
    print(f'AP: {ap:.2f}, FPR95: {fpr95:.2f}')

    # Example 3: loading the scores from disk
    ap, fpr95 = ade_ood_eval_with_scores_from_disk(os.path.expandvars('$SCORES_PATH'))
    print(f'AP: {ap:.2f}, FPR95: {fpr95:.2f}')

