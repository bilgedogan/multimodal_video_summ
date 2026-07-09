"""Compare training runs (different fusion_type configs) side by side.

Reads TensorBoard event files written under each run's Summaries/<exp>/<dataset>/<tag>/
directory (bound to the trainer logger in train.py) and plots the key metrics
(train_loss, val_kTau/sRho/f1, per-modality fusion weights, and RL reward/policy_loss
when present) for every discovered run on the same axes.

Usage:
    python visualize.py --logdir Summaries --out compare.png
    python visualize.py --logdir Summaries/exp_2 --out compare.png
"""
import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from utils.configs import str2bool

METRICS = [
    'train_loss_epoch',
    'val_kTau',
    'val_sRho',
    'val_f1',
    'val_w_text',
    'val_w_audio',
    'val_w_visual',
    'reward_tau_epoch',
    'policy_loss_epoch',
]


def find_runs(logdir):
    """A run = a directory that (directly or in a subdir) contains tfevents files."""
    event_files = glob.glob(os.path.join(logdir, '**', 'events.out.tfevents.*'), recursive=True)
    run_dirs = sorted({os.path.dirname(f) for f in event_files})
    return run_dirs


def run_label(run_dir):
    config_path = os.path.join(run_dir, 'configuration.txt')
    if os.path.isfile(config_path):
        text = open(config_path).read()
        m = re.search(r"'fusion_type':\s*'([^']+)'", text)
        tag = re.search(r"'tag':\s*'([^']+)'", text)
        if m:
            label = m.group(1)
            if tag:
                label += f' ({tag.group(1)})'
            return label
    return os.path.relpath(run_dir)


def load_scalars(run_dir, tag):
    ea = EventAccumulator(run_dir, size_guidance={'scalars': 0})
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        return None, None
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='Summaries', help='root dir to search for runs')
    parser.add_argument('--out', type=str, default='compare.png', help='output png path')
    parser.add_argument('--wandb', type=str2bool, default=True, help='log the comparison plot to wandb')
    parser.add_argument('--wandb_project', type=str, default='multimodal-video-summ')
    args = parser.parse_args()

    run_dirs = find_runs(args.logdir)
    if not run_dirs:
        print(f'no tfevents found under {args.logdir}')
        return

    n_metrics = len(METRICS)
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten()

    for run_dir in run_dirs:
        label = run_label(run_dir)
        for ax, metric in zip(axes, METRICS):
            steps, values = load_scalars(run_dir, metric)
            if steps is None:
                continue
            ax.plot(steps, values, label=label)

    for ax, metric in zip(axes, METRICS):
        ax.set_title(metric)
        ax.set_xlabel('step')
        ax.grid(alpha=0.3)
        if ax.lines:
            ax.legend(fontsize=8)

    for ax in axes[n_metrics:]:
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f'saved {args.out} ({len(run_dirs)} runs found)')
    for run_dir in run_dirs:
        print(f'  - {run_label(run_dir)}  <- {run_dir}')

    if args.wandb:
        run = wandb.init(project=args.wandb_project, job_type='visualize', name=os.path.basename(args.out))
        run.log({'comparison': wandb.Image(args.out)})
        artifact = wandb.Artifact('comparison-plot', type='visualization')
        artifact.add_file(args.out)
        run.log_artifact(artifact)
        run.finish()


if __name__ == '__main__':
    main()
