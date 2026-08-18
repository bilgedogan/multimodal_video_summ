"""Wilcoxon signed-rank tests (+ Holm-Bonferroni) of experiments vs one baseline.

Reads the per-split lines written by test_splits.py
    summe_split0: kTau=0.184 sRho=0.206 f1=0.438
from Summaries/<exp>/seed_<seed>/<dataset>_<weights>_results.txt.

Samples are paired by the same key in both experiments:
    --granularity split  (default) one sample per (seed, split)  -> 5 seeds x 5 splits = 25
    --granularity seed              one sample per seed (mean over splits) -> 5

Every --exps entry is compared against --baseline. Holm-Bonferroni is applied
over that whole group of comparisons, separately for each dataset and metric.

Usage:
    python wilcoxon_test.py --baseline exp_0 --exps exp_1,exp_2,exp_3 --tag vs_exp_0
"""
import argparse
import glob
import os
import re

import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

METRICS = ['kTau', 'sRho', 'f1']
DATASETS = ['summe', 'tvsum']

# "summe_split0: kTau=0.184 sRho=0.206 f1=0.438"
SPLIT_RE = re.compile(
    r'^\w+_split(\d+)\s*:\s*' + r'\s+'.join(r'{}=([-\d.]+)'.format(m) for m in METRICS)
)
# "kTau: 0.240 +/- 0.041"
SUMMARY_RE = {m: re.compile(r'^{}\s*:\s*([-\d.]+)\s*\+/-'.format(m)) for m in METRICS}


def parse_result_file(path):
    """Return (per_split, summary) parsed from one *_results.txt."""
    per_split, summary = {}, {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            match = SPLIT_RE.match(line)
            if match:
                split = int(match.group(1))
                per_split[split] = {m: float(match.group(i + 2)) for i, m in enumerate(METRICS)}
                continue
            for metric, pattern in SUMMARY_RE.items():
                match = pattern.match(line)
                if match:
                    summary[metric] = float(match.group(1))
    return per_split, summary


def collect_scores(result_dir, exp_name, dataset, weights, granularity):
    """Return dict sample_key -> {metric: value}; key is (seed, split) or seed."""
    samples = {}
    for seed_dir in sorted(glob.glob(os.path.join(result_dir, exp_name, 'seed_*'))):
        seed = os.path.basename(seed_dir).replace('seed_', '')
        result_file = os.path.join(seed_dir, '{}_{}_results.txt'.format(dataset, weights))
        if not os.path.isfile(result_file):
            continue
        per_split, summary = parse_result_file(result_file)
        if granularity == 'split':
            for split, scores in per_split.items():
                samples[(seed, split)] = scores
        elif summary:
            samples[seed] = summary
    return samples


def sample_sort_key(key):
    if isinstance(key, tuple):
        seed, split = key
        return (len(seed), seed, split)
    return (len(key), key)


def compare(samples_a, samples_b, metric, alternative):
    """Paired test of exp (a) against baseline (b). None if fewer than 2 pairs."""
    keys = sorted(
        (k for k in set(samples_a) & set(samples_b)
         if metric in samples_a[k] and metric in samples_b[k]),
        key=sample_sort_key,
    )
    if len(keys) < 2:
        return None
    vals_a = np.array([samples_a[k][metric] for k in keys])
    vals_b = np.array([samples_b[k][metric] for k in keys])

    if np.all(vals_a == vals_b):
        w_p = 1.0
    else:
        _, w_p = wilcoxon(vals_a, vals_b, alternative=alternative)

    t_stat, t_p = ttest_rel(vals_a, vals_b)
    if alternative == 'greater':
        t_p = t_p / 2 if t_stat > 0 else 1 - t_p / 2
    elif alternative == 'less':
        t_p = t_p / 2 if t_stat < 0 else 1 - t_p / 2

    return {'n': len(keys),
            'mean_a': vals_a.mean(), 'std_a': vals_a.std(),
            'mean_b': vals_b.mean(), 'std_b': vals_b.std(),
            'w_p': w_p, 't_p': t_p}


def run(result_dir, weights, baseline, exps, tag, granularity, alternative, alpha, out_dir):
    unit = '(seed, split)' if granularity == 'split' else 'seed'
    missing = [e for e in [baseline] + exps
               if not glob.glob(os.path.join(result_dir, e, 'seed_*'))]
    exps = [e for e in exps if e not in missing]
    if missing:
        print('[{}] not found under {}, skipped: {}'.format(tag, result_dir, ', '.join(missing)))
    if baseline in missing or not exps:
        print('[{}] nothing to test.'.format(tag))
        return

    for dataset in DATASETS:
        base_samples = collect_scores(result_dir, baseline, dataset, weights, granularity)
        if not base_samples:
            print('[{}] {}: baseline {} has no {} results, skipped'.format(
                tag, dataset, baseline, weights))
            continue

        per_exp = {}
        for exp in exps:
            samples = collect_scores(result_dir, exp, dataset, weights, granularity)
            if samples:
                per_exp[exp] = samples
        if not per_exp:
            print('[{}] {}: no experiment has {} results, skipped'.format(tag, dataset, weights))
            continue

        lines = [
            'Wilcoxon signed-rank (paired by {}) + Holm-Bonferroni'.format(unit),
            'group={}  baseline={}  dataset={}  checkpoint selection={}'.format(
                tag, baseline, dataset, weights),
            'compared: {}'.format(', '.join(per_exp)),
            'alternative={}  alpha={}  (Holm family = this group, per metric)'.format(
                alternative, alpha),
            '',
        ]

        for metric in METRICS:
            names, results = [], []
            for exp, samples in per_exp.items():
                res = compare(samples, base_samples, metric, alternative)
                if res is None:
                    lines.append('{}: {} vs {}: fewer than 2 paired samples, skipped'.format(
                        metric, exp, baseline))
                    continue
                names.append(exp)
                results.append(res)
            if not results:
                lines.append('{}: no testable comparisons'.format(metric))
                lines.append('')
                continue

            reject, p_corrected, _, _ = multipletests(
                [r['w_p'] for r in results], alpha=alpha, method='holm')

            n_values = sorted({r['n'] for r in results})
            lines.append('--- {} (n={}, baseline {}={:.4f}+/-{:.4f}) ---'.format(
                metric, ', '.join(str(n) for n in n_values), baseline,
                results[0]['mean_b'], results[0]['std_b']))
            lines.append('{:<16} | {:<17} | {:<11} | {:<11} | {:<10} | {}'.format(
                'experiment', 'mean', 'uncorr p', 'Holm p', 't-test p', 'significant?'))
            lines.append('-' * 92)
            for name, res, rej, p_corr in zip(names, results, reject, p_corrected):
                if rej:
                    verdict = 'Yes ({} better)'.format(
                        name if res['mean_a'] > res['mean_b'] else baseline)
                else:
                    verdict = 'No'
                lines.append('{:<16} | {:<17} | {:<11.5f} | {:<11.5f} | {:<10.5f} | {}'.format(
                    name, '{:.4f}+/-{:.4f}'.format(res['mean_a'], res['std_a']),
                    res['w_p'], p_corr, res['t_p'], verdict))
            lines.append('')

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, 'wilcoxon_{}_{}_{}_{}.txt'.format(
            dataset, weights, granularity, tag))
        with open(out_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print('\n'.join(lines))
        print('Saved: {}'.format(out_path))
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='Summaries')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='where to write reports (default <result_dir>/stats)')
    parser.add_argument('--weights', type=str, default='tau',
                        help="which checkpoint-selection results to read, 'tau' or 'rho'")
    parser.add_argument('--baseline', type=str, required=True)
    parser.add_argument('--exps', type=str, required=True,
                        help='comma-separated experiments to test against the baseline')
    parser.add_argument('--tag', type=str, default=None, help='name used in the output filename')
    parser.add_argument('--granularity', type=str, default='split', choices=['split', 'seed'])
    parser.add_argument('--alternative', type=str, default='two-sided',
                        choices=['two-sided', 'greater', 'less'])
    parser.add_argument('--alpha', type=float, default=0.05)
    args = parser.parse_args()

    exps = [e.strip() for e in args.exps.split(',') if e.strip()]
    tag = args.tag or 'vs_{}'.format(args.baseline)
    out_dir = args.out_dir or os.path.join(args.result_dir, 'stats')
    run(args.result_dir, args.weights, args.baseline, exps, tag,
        args.granularity, args.alternative, args.alpha, out_dir)
