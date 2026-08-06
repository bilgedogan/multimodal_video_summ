"""Pairwise paired t-tests across seeds for all experiments in Summaries/.

For each dataset and metric (kTau, sRho, f1), gathers the per-seed mean
score (mean over 5 splits, as already computed by test_splits.py) for every
exp_name found under Summaries/, then runs a paired t-test (scipy.stats.ttest_rel)
between every pair of experiments using seeds common to both.

Usage:
    python ttest_seeds.py [--result_dir Summaries] [--weights tau]
"""
import argparse
import glob
import os
import re
from itertools import combinations

import numpy as np
from scipy.stats import ttest_rel

METRICS = ['kTau', 'sRho', 'f1']
DATASETS = ['summe', 'tvsum']

SUMMARY_RE = {
    m: re.compile(r'^{}\s*:\s*([-\d.]+)\s*\+/-'.format(m)) for m in METRICS
}


def parse_result_file(path):
    """Return dict metric -> mean value from a *_results.txt summary section."""
    scores = {}
    with open(path) as f:
        for line in f:
            for metric, pattern in SUMMARY_RE.items():
                match = pattern.match(line.strip())
                if match:
                    scores[metric] = float(match.group(1))
    print(scores)
    return scores


def collect_scores(result_dir, exp_name, dataset, weights):
    """Return dict seed -> {metric: value} for one exp/dataset."""
    per_seed = {}
    seed_dirs = sorted(glob.glob(os.path.join(result_dir, exp_name, 'seed_*')))
    for seed_dir in seed_dirs:
        seed = os.path.basename(seed_dir).replace('seed_', '')
        result_file = os.path.join(seed_dir, '{}_{}_results.txt'.format(dataset, weights))
        if not os.path.isfile(result_file):
            continue
        scores = parse_result_file(result_file)
        if scores:
            per_seed[seed] = scores
    return per_seed


def find_exp_names(result_dir):
    names = []
    for path in sorted(glob.glob(os.path.join(result_dir, 'exp_*'))):
        if os.path.isdir(path) and glob.glob(os.path.join(path, 'seed_*')):
            names.append(os.path.basename(path))
    return names


def run_ttests(result_dir, weights):
    exp_names = find_exp_names(result_dir)
    if len(exp_names) < 2:
        print('Need at least 2 experiments under {}, found: {}'.format(result_dir, exp_names))
        return

    for dataset in DATASETS:
        per_exp_scores = {exp: collect_scores(result_dir, exp, dataset, weights) for exp in exp_names}
        per_exp_scores = {exp: s for exp, s in per_exp_scores.items() if s}
        if len(per_exp_scores) < 2:
            print('Skipping {}: fewer than 2 experiments have results ({})'.format(dataset, weights))
            continue

        lines = ['T-test (paired, by seed) — dataset={}, checkpoint selection={}'.format(dataset, weights), '']

        for exp_a, exp_b in combinations(sorted(per_exp_scores), 2):
            seeds_a, seeds_b = per_exp_scores[exp_a], per_exp_scores[exp_b]
            common_seeds = sorted(set(seeds_a) & set(seeds_b), key=lambda s: (len(s), s))
            if len(common_seeds) < 2:
                lines.append('{} vs {}: fewer than 2 common seeds ({}), skipped'.format(
                    exp_a, exp_b, common_seeds))
                continue

            lines.append('{} vs {}  (n={} seeds: {})'.format(exp_a, exp_b, len(common_seeds), ', '.join(common_seeds)))
            for metric in METRICS:
                vals_a = np.array([seeds_a[s][metric] for s in common_seeds if metric in seeds_a[s] and metric in seeds_b[s]])
                vals_b = np.array([seeds_b[s][metric] for s in common_seeds if metric in seeds_a[s] and metric in seeds_b[s]])
                if len(vals_a) < 2:
                    lines.append('  {}: not enough paired samples, skipped'.format(metric))
                    continue
                stat, pval = ttest_rel(vals_a, vals_b)
                sig = '*' if pval < 0.05 else ''
                lines.append('  {}: {}={:.4f}+/-{:.4f} vs {}={:.4f}+/-{:.4f}  t={:.3f} p={:.4f}{}'.format(
                    metric, exp_a, vals_a.mean(), vals_a.std(), exp_b, vals_b.mean(), vals_b.std(), stat, pval, sig))
                if pval < 0.05:
                    better = exp_a if vals_a.mean() > vals_b.mean() else exp_b
                    lines.append('    -> {} better (significant)'.format(better))
                else:
                    lines.append('    -> no significant difference')
            lines.append('')

        out_path = os.path.join(result_dir, 'ttest_{}_{}.txt'.format(dataset, weights))
        with open(out_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print('\n'.join(lines))
        print('Saved: {}'.format(out_path))
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='Summaries')
    parser.add_argument('--weights', type=str, default='tau', help="which checkpoint-selection results to read, e.g. 'tau' or 'rho'")
    args = parser.parse_args()
    run_ttests(args.result_dir, args.weights)
