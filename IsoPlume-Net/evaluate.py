import os
import argparse

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import beta
from skimage import measure

from model import MultiScaleAttentionUNet
from data import MultiModalPlumeDataset, MultiModalDataProcessor


SIZE_BINS = {'small': (50, 500), 'medium': (500, 3000), 'large': (3000, float('inf'))}


class PlumeEvaluator:
    """
    Plume-level and separation metrics.
    Connected components smaller than min_plume_size pixels are ignored.
    Predicted and ground-truth plumes are matched one-to-one with the
    Hungarian algorithm on IoU. 
    """

    def __init__(self, min_plume_size=50, iou_threshold=0.3,
                 quality_iou_threshold=0.5, overlap_fraction=0.3):
        self.min_plume_size = min_plume_size
        self.iou_threshold = iou_threshold
        self.quality_iou_threshold = quality_iou_threshold
        self.overlap_fraction = overlap_fraction

    def extract_components(self, mask):
        labeled = measure.label(mask, connectivity=2)
        return [{'mask': labeled == r.label, 'size': r.area}
                for r in measure.regionprops(labeled) if r.area >= self.min_plume_size]

    @staticmethod
    def size_bin(area):
        for name, (lo, hi) in SIZE_BINS.items():
            if lo <= area < hi:
                return name
        return 'large'

    @staticmethod
    def compute_iou(m1, m2):
        union = (m1 | m2).sum()
        return (m1 & m2).sum() / union if union > 0 else 0.0

    def match_components(self, gt_c, pred_c):
        if not gt_c or not pred_c:
            return [], len(gt_c), len(pred_c)
        mat = np.array([[self.compute_iou(g['mask'], p['mask']) for p in pred_c] for g in gt_c])
        rows, cols = linear_sum_assignment(-mat)
        matches = [(i, j, mat[i, j]) for i, j in zip(rows, cols) if mat[i, j] >= self.iou_threshold]
        return matches, len(gt_c) - len(matches), len(pred_c) - len(matches)

    def count_bridging(self, pred_c, gt_c):
        """A predicted component overlaps two or more ground-truth plumes."""
        n = 0
        for pc in pred_c:
            ov = sum(1 for gc in gt_c
                     if (pc['mask'] & gc['mask']).sum() > self.overlap_fraction * gc['size'])
            if ov >= 2:
                n += ov - 1
        return n

    def count_fragmentation(self, pred_c, gt_c):
        """A ground-truth plume is covered by two or more predicted components."""
        n = 0
        for gc in gt_c:
            cov = sum(1 for pc in pred_c
                      if (pc['mask'] & gc['mask']).sum() > self.overlap_fraction * gc['size'])
            if cov >= 2:
                n += 1
        return n

    def evaluate_single(self, pred_binary, gt_mask):
        pred_binary = pred_binary.astype(bool)
        gt_mask = gt_mask.astype(bool)
        gt_c = self.extract_components(gt_mask)
        pred_c = self.extract_components(pred_binary)

        matches, n_missed, n_false_pos = self.match_components(gt_c, pred_c)
        ious = [iou for _, _, iou in matches]
        iou_of_gt = {i: iou for i, _, iou in matches}

        size_recall, size_iou = {}, {}
        for bn in SIZE_BINS:
            gt_in = [i for i, c in enumerate(gt_c) if self.size_bin(c['size']) == bn]
            if not gt_in:
                continue
            matched = [iou_of_gt[i] for i in gt_in if i in iou_of_gt]
            size_recall[bn] = len(matched) / len(gt_in)
            size_iou[bn] = np.mean(matched) if matched else 0.0

        return {
            'tp': int((pred_binary & gt_mask).sum()),
            'fp': int((pred_binary & ~gt_mask).sum()),
            'fn': int((~pred_binary & gt_mask).sum()),
            'n_gt': len(gt_c),
            'n_pred': len(pred_c),
            'n_matched': len(matches),
            'n_hq_matched': sum(1 for v in ious if v >= self.quality_iou_threshold),
            'n_missed': n_missed,
            'n_false_pos': n_false_pos,
            'n_bridged': self.count_bridging(pred_c, gt_c),
            'n_fragmented': self.count_fragmentation(pred_c, gt_c),
            'per_plume_ious': ious,
            'size_recall': size_recall,
            'size_iou': size_iou,
        }


def summary_metrics(results):
    """Table 2 metrics, pooled over the given samples."""
    tp = sum(r['tp'] for r in results)
    fp = sum(r['fp'] for r in results)
    fn = sum(r['fn'] for r in results)
    n_gt = sum(r['n_gt'] for r in results)
    n_pred = sum(r['n_pred'] for r in results)
    n_matched = sum(r['n_matched'] for r in results)
    ious = [v for r in results for v in r['per_plume_ious']]

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    plume_recall = n_matched / n_gt if n_gt > 0 else 0.0
    plume_precision = n_matched / n_pred if n_pred > 0 else 0.0

    return {
        'Pixel IoU': tp / (tp + fp + fn + 1e-6),
        'Pixel Precision': precision,
        'Pixel Recall': recall,
        'Pixel F1': 2 * precision * recall / (precision + recall + 1e-6),
        'Plume Recall': plume_recall,
        'Plume Precision': plume_precision,
        'Plume F1': 2 * plume_precision * plume_recall / (plume_precision + plume_recall + 1e-6),
        'Median plume IoU': float(np.median(ious)) if ious else 0.0,
        'Bridging rate': sum(r['n_bridged'] for r in results) / n_gt if n_gt > 0 else 0.0,
        'Fragmentation rate': sum(r['n_fragmented'] for r in results) / n_gt if n_gt > 0 else 0.0,
    }


def bootstrap_ci(results, n_boot=1000, seed=42):
    """95% bootstrap confidence intervals, resampling at the sample level."""
    rng = np.random.RandomState(seed)
    n = len(results)
    samples = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        samples.append(summary_metrics([results[i] for i in idx]))
    return {k: (np.percentile([s[k] for s in samples], 2.5),
                np.percentile([s[k] for s in samples], 97.5)) for k in samples[0]}


def binomial_ci(k, n, alpha=0.05):
    """Exact (Clopper-Pearson) binomial confidence interval."""
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = MultiScaleAttentionUNet(
        n_channels_co=1, n_channels_fire=1, n_channels_rgb=3, n_classes=1,
        use_attention=True, use_fusion=True,
        use_fire=True, use_rgb1=True, use_false_rgb=True
    )
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval(), ckpt


def load_stats(ckpt, stats_path):
    if stats_path is not None:
        co_stats, fire_stats, rgb_stats = np.load(stats_path, allow_pickle=True)
    elif 'data_stats' in ckpt:
        s = ckpt['data_stats']
        co_stats, fire_stats, rgb_stats = s['co'], s['fire'], s['rgb']
    else:
        raise ValueError("No normalisation statistics found. Please provide --stats.")
    return co_stats, fire_stats, rgb_stats


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    model, ckpt = load_model(args.checkpoint, device)

    processor = MultiModalDataProcessor(image_size=(256, 256))
    processor.co_stats, processor.fire_stats, processor.rgb_stats = load_stats(ckpt, args.stats)

    file_lists = np.load(args.test_list, allow_pickle=True).item()
    dataset = MultiModalPlumeDataset(file_lists, processor, augment=False, normalize=True)

    evaluator = PlumeEvaluator(min_plume_size=50, iou_threshold=0.3,
                               quality_iou_threshold=0.5, overlap_fraction=0.3)

    results = []
    for idx in range(len(dataset)):
        co, fire, rgb1, false_rgb, mask = dataset[idx]
        with torch.no_grad():
            out = model(co[None].to(device), fire[None].to(device),
                        rgb1[None].to(device), false_rgb[None].to(device))
        pred_binary = (torch.sigmoid(out)[0, 0].cpu().numpy() > args.threshold)
        results.append(evaluator.evaluate_single(pred_binary, mask[0].numpy() > 0.5))

    metrics = summary_metrics(results)
    ci = bootstrap_ci(results, n_boot=1000, seed=args.seed)

    n_gt = sum(r['n_gt'] for r in results)
    ci['Fragmentation rate'] = binomial_ci(sum(r['n_fragmented'] for r in results), n_gt)

    print(f"Test samples: {len(results)}, ground-truth plumes: {n_gt}")
    rows = []
    for k, v in metrics.items():
        lo, hi = ci[k]
        print(f"{k:<20} {v:.3f} [{lo:.3f}, {hi:.3f}]")
        rows.append({'Metric': k, 'Value': v, 'CI_low': lo, 'CI_high': hi})

    for bn in SIZE_BINS:
        rec = [r['size_recall'][bn] for r in results if bn in r['size_recall']]
        iou = [r['size_iou'][bn] for r in results if bn in r['size_iou']]
        if rec:
            print(f"{bn.capitalize():<7} recall {np.mean(rec):.3f}, mean plume IoU {np.mean(iou):.3f}")

    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, 'test_metrics.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate IsoPlume-Net on the independent test set')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--test_list', required=True, help='Path to test_file_lists.npy')
    parser.add_argument('--stats', default=None,
                        help='Path to data_stats.npy, if not stored in the checkpoint')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    main(parser.parse_args())