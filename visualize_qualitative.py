# -*- coding: utf-8 -*-
"""Qualitative per-video figures for a trained LLMVS checkpoint.

For every video in the evaluation set one PNG (and optionally PDF) is written with
three stacked rows:

    row 1 : key frames sampled from the video (green = predicted peaks, pink = valleys)
    row 2 : predicted importance score vs. ground-truth score over time
    row 3 : per-frame modality weights w_text / w_audio / w_visual

Example
-------
python visualize_qualitative.py \
    --run_dir Summaries/exp_4/seed_1112/summe/summe_split4 \
    --weights rho --out_dir figures/exp_4_split4
"""

import os
import re
import ast
import csv
import glob
import json
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import ConnectionPatch

import imageio.v2 as iio

from utils.configs import str2bool
from utils.generate_summary import generate_summary
from utils.evaluation_metrics import evaluate_summary


# ----------------------------------------------------------------------------- style
C_PRED = '#E2662C'      # predicted importance score
C_GT = '#4878A8'        # ground-truth importance score
C_W = {'text': '#6A3D9A', 'audio': '#00846B', 'visual': '#C79200'}
C_BAND = {'high': '#B7E1A1', 'low': '#F0C0DC'}
C_EDGE = {'high': '#5FA346', 'low': '#C77BAA'}

DEFAULT_VIDEO_DIRS = {
    'summe': '/home/bilge/Documents/Video_Summ/feature_extraction/DataSet/SumMe/videos',
    'tvsum': '/home/bilge/Documents/Video_Summ/feature_extraction/DataSet/TVSum',
}

CONFIG_DEFAULTS = dict(
    reduced_dim=2048, num_heads=2, num_layers=3, audio_dim=2048, visual_dim=768,
    weight_gen_dim=256, fusion_type='global_weight', diffusion=False, diffusion_steps=20,
    diffusion_weight=0.1, diversity_weight=0.0, diversity_lambda=20, rl_weight=0.1,
    baseline_momentum=0.9, tr_val_ts=False, dataset='summe', split_idx=0,
    pt_path='llama_emb/summe_sum/', audio_path=None, visual_path=None,
)


class Cfg(object):
    """Plain attribute bag - unlike utils.configs.Config it touches no files."""

    def __init__(self, d):
        self.__dict__.update(d)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __repr__(self):
        return 'Cfg({})'.format(self.__dict__)


# ------------------------------------------------------------------------- run config
def read_train_config(run_dir):
    """Parse the configuration.txt written next to the checkpoints."""
    path = os.path.join(run_dir, 'configuration.txt')
    if not os.path.isfile(path):
        return {}
    with open(path, 'r') as f:
        txt = f.read()
    try:
        body = txt[txt.index('{'):txt.rindex('}') + 1]
        body = re.sub(r"'device':\s*device\([^)]*\),\s*", '', body)
        return ast.literal_eval(body)
    except Exception as e:                                    # noqa: BLE001
        print('[warn] could not parse {}: {}'.format(path, e))
        return {}


def find_checkpoint(run_dir, weights):
    """`weights` is either 'rho' / 'tau' or a direct path to a .ckpt."""
    if weights.endswith('.ckpt'):
        return weights
    sub = 'best_rho_model' if weights == 'rho' else 'best_tau_model'
    hits = sorted(glob.glob(os.path.join(run_dir, sub, '*.ckpt')))
    if not hits:
        raise FileNotFoundError('no checkpoint under {}'.format(os.path.join(run_dir, sub)))
    return hits[-1]


# ------------------------------------------------------------------------ key frames
def select_keyframes(score, n_high, n_low, min_sep_frac, edge_frac=0.0):
    """Greedy peak / valley picking with a minimum temporal separation.

    `edge_frac` keeps the selection away from the very start and end of the video,
    where intro / outro cards are usually black and make useless thumbnails.
    """
    n = len(score)
    min_sep = max(1, int(round(n * min_sep_frac)))
    lo = int(round(n * edge_frac))
    hi = n - 1 - lo
    ok = lambda i: lo <= i <= hi                              # noqa: E731

    high = []
    for i in np.argsort(-score):
        if ok(i) and all(abs(int(i) - j) >= min_sep for j in high):
            high.append(int(i))
        if len(high) == n_high:
            break

    low = []
    for i in np.argsort(score):
        if ok(i) and all(abs(int(i) - j) >= min_sep for j in high + low):
            low.append(int(i))
        if len(low) == n_low:
            break

    picked = [(i, 'high') for i in high] + [(i, 'low') for i in low]
    return sorted(picked, key=lambda p: p[0])


def build_name_map(dataset, name_map_path=None):
    """TVSum's h5 has no `video_name` field: video_i must be mapped to a YouTube id.

    The mapping is the order of the videos in ydata-tvsum50-anno.tsv, which is also
    what utils/evaluation_metrics.py assumes when it does int(video_name.split('_')[-1]).
    """
    if 'tvsum' not in dataset:
        return {}

    for path in ([name_map_path] if name_map_path else []) + [
            'dataset/tvsum_mapped_video_names.json',
            '/home/bilge/Documents/Video_Summ/feature_extraction/name_mapping/'
            'tvsum_mapped_video_names.json']:
        if path and os.path.isfile(path):
            raw = json.load(open(path))
            return {v: k for k, v in raw.items()}      # {'video_1': 'AwmHb44_ouw'}

    tsv = 'TVSum/ydata-tvsum50-anno.tsv'
    if os.path.isfile(tsv):
        with open(tsv) as f:
            ids = list(dict.fromkeys(row.split('\t')[0] for row in f if row.strip()))
        return {'video_{}'.format(i + 1): vid for i, vid in enumerate(ids)}

    print('[warn] no TVSum name mapping found - key frames will be missing')
    return {}


def resolve_video_path(video_dir, video_filename, video_name, name_map=None):
    """`video_filename` comes from the h5 as the string "b'Fire Domino'"."""
    cands = []
    if video_filename:
        stem = video_filename[2:-1] if video_filename.startswith("b'") else video_filename
        cands += [stem.replace(' ', '_'), stem]
    if name_map and video_name in name_map:
        cands.append(name_map[video_name])
    cands.append(video_name)
    for stem in cands:
        for ext in ('.mp4', '.webm', '.avi', '.mkv'):
            p = os.path.join(video_dir, stem + ext)
            if os.path.isfile(p):
                return p
    return None


def grab_frames(video_path, frame_ids, default_fps=30.0):
    """Read the requested original-video frame indices; None where unavailable.

    Also returns the container's real frame rate - TVSum clips are not all 30 fps.
    """
    out = [None] * len(frame_ids)
    if video_path is None:
        return out, default_fps
    try:
        reader = iio.get_reader(video_path)
    except Exception as e:                                    # noqa: BLE001
        print('[warn] cannot open {}: {}'.format(video_path, e))
        return out, default_fps
    fps = float(reader.get_meta_data().get('fps') or default_fps)
    for k, fid in enumerate(frame_ids):
        for cand in (int(fid), max(0, int(fid) - 1), max(0, int(fid) - 15)):
            try:
                out[k] = reader.get_data(cand)
                break
            except Exception:                                 # noqa: BLE001
                continue
    reader.close()
    return out, fps


# ----------------------------------------------------------------------------- helpers
def minmax(v):
    v = np.asarray(v, dtype=np.float64)
    lo, hi = v.min(), v.max()
    return np.zeros_like(v) if hi - lo < 1e-12 else (v - lo) / (hi - lo)


def moving_average(v, k):
    if k <= 1:
        return v
    pad = k // 2
    padded = np.pad(np.asarray(v, dtype=np.float64), (pad, pad), mode='edge')
    return np.convolve(padded, np.ones(k) / k, mode='valid')[:len(v)]


# ------------------------------------------------------------------------------ figure
def make_figure(out_base, video_name, title, score, gt, w, picks, fps, frames,
                keyframes, args):
    n = len(score)
    t = np.arange(n)
    k = len(keyframes)

    s_plot = minmax(score) if args.normalize else np.asarray(score, dtype=np.float64)
    g_plot = minmax(gt) if args.normalize else np.asarray(gt, dtype=np.float64)
    s_plot = moving_average(s_plot, args.smooth)
    g_plot = moving_average(g_plot, args.smooth)

    thumb_h = 2.35 if args.captions else 1.95
    fig_h = thumb_h + 3.5
    fig = plt.figure(figsize=(args.fig_width, fig_h))
    gs = GridSpec(3, 1, figure=fig,
                  height_ratios=[thumb_h, 1.65, 1.35],
                  hspace=0.30, left=0.055, right=0.985, top=0.955, bottom=0.085)

    # --- row 1: key frames -------------------------------------------------------
    gs_thumb = GridSpecFromSubplotSpec(1, k, subplot_spec=gs[0], wspace=0.06)
    thumb_axes = []
    for j, (idx, kind) in enumerate(keyframes):
        ax = fig.add_subplot(gs_thumb[0, j])
        img = frames[j]
        if img is None:
            ax.text(0.5, 0.5, 'frame\nunavailable', ha='center', va='center',
                    fontsize=7, color='#999999', transform=ax.transAxes)
            ax.set_facecolor('#F2F2F2')
        else:
            ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(C_EDGE[kind])
            sp.set_linewidth(1.6)

        sec = picks[idx] / fps if fps else 0.0
        cap = 't = {:.1f}s   $\\hat{{s}}$ = {:.2f}'.format(sec, s_plot[idx])
        if args.captions:
            extra = args.captions.get(video_name, {}).get(str(int(picks[idx])))
            if extra:
                cap = extra + '\n' + cap
        ax.set_xlabel(cap, fontsize=7.0, labelpad=3, color='#333333')
        thumb_axes.append(ax)

    # --- row 2: importance scores ------------------------------------------------
    ax_s = fig.add_subplot(gs[1])
    ax_s.plot(t, g_plot, color=C_GT, lw=1.25, label='Ground truth', zorder=3)
    ax_s.plot(t, s_plot, color=C_PRED, lw=1.35, label='Ours (predicted)', zorder=4)
    ax_s.set_ylabel('Importance\nscore $s$', fontsize=9)
    ax_s.set_xlim(0, n - 1)
    ax_s.set_ylim(-0.05, 1.12 if args.normalize else None)
    ax_s.legend(loc='upper right', fontsize=7.5, ncol=2, frameon=True,
                framealpha=0.9, borderpad=0.35, handlelength=1.6)

    # --- row 3: modality weights -------------------------------------------------
    ax_w = fig.add_subplot(gs[2], sharex=ax_s)
    # w_text is drawn last so it stays visible where the Dirichlet alphas saturate
    # at their floor of 1 and two modality weights coincide to ~1e-5.
    zorders = {'visual': 3, 'audio': 4, 'text': 5}
    for j, name in enumerate(('text', 'audio', 'visual')):
        series = moving_average(w[:, j], args.smooth)
        ax_w.plot(t, series, color=C_W[name], lw=1.35, zorder=zorders[name],
                  label='$w_{{{}}}$  ({:.3f})'.format(name, float(w[:, j].mean())))

    # flag coincident curves explicitly - otherwise the reader just sees a missing line
    coincident = ['$w_{{{}}} \\equiv w_{{{}}}$'.format(a, b)
                  for a, b, i, j in (('text', 'audio', 0, 1), ('text', 'visual', 0, 2),
                                     ('audio', 'visual', 1, 2))
                  if np.abs(w[:, i] - w[:, j]).max() < 1e-3]
    if coincident:
        ax_w.text(0.008, 0.94, '  '.join(coincident) + '  (< $10^{-3}$)',
                  transform=ax_w.transAxes, fontsize=7.5, va='top', color='#555555')
    ax_w.axhline(1.0 / 3.0, color='#AAAAAA', lw=0.7, ls=':', zorder=1)
    ax_w.set_ylabel('Modality\nweight $w$', fontsize=9)
    ax_w.set_xlabel('Time step $t$', fontsize=9)
    ax_w.legend(loc='upper right', fontsize=7.5, ncol=3, frameon=True,
                framealpha=0.9, borderpad=0.35, handlelength=1.6)
    wmin, wmax = float(w.min()), float(w.max())
    rng = max(1e-3, wmax - wmin)
    ax_w.set_ylim(max(-0.02, wmin - 0.08 * rng), min(1.02, wmax + 0.42 * rng))

    for ax in (ax_s, ax_w):
        ax.grid(True, axis='y', color='#E6E6E6', lw=0.6, zorder=0)
        ax.tick_params(labelsize=8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#666666')
            ax.spines[side].set_linewidth(0.8)

    # --- highlight bands + leader lines -----------------------------------------
    half = max(1.0, n * 0.011)
    y_top = ax_s.get_ylim()[1]
    for j, (idx, kind) in enumerate(keyframes):
        for ax in (ax_s, ax_w):
            ax.axvspan(idx - half, idx + half, color=C_BAND[kind], alpha=0.55,
                       lw=0, zorder=2)
        con = ConnectionPatch(xyA=(0.5, -0.02), coordsA=thumb_axes[j].transAxes,
                              xyB=(idx, y_top), coordsB=ax_s.transData,
                              color=C_EDGE[kind], lw=0.9, ls=(0, (4, 3)), alpha=0.9)
        con.set_zorder(1)
        fig.add_artist(con)

    fig.suptitle(title, fontsize=10.5, y=0.995)

    fig.savefig(out_base + '.png', dpi=args.dpi)
    if args.pdf:
        fig.savefig(out_base + '.pdf')
    plt.close(fig)


# --------------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', type=str,
                   default='Summaries/exp_4/seed_1112/summe/summe_split4',
                   help='training run directory holding configuration.txt + best_*_model/')
    p.add_argument('--weights', type=str, default='rho',
                   help="'rho', 'tau', or a direct path to a .ckpt")
    p.add_argument('--out_dir', type=str, default=None,
                   help='where the PNGs go (default: <run_dir>/qualitative)')
    p.add_argument('--mode', type=str, default='test', choices=['test', 'val', 'train'],
                   help="evaluation set to render; with tr_val_ts=False 'test' IS the "
                        'set used for validation during training')
    p.add_argument('--videos_dir', type=str, default=None)
    p.add_argument('--name_map', type=str, default=None,
                   help='TVSum only: json mapping YouTube ids to video_i keys')
    p.add_argument('--captions_json', type=str, default=None,
                   help='optional {video_name: {frame_idx: caption}} json for the top row')

    # overrides (otherwise taken from configuration.txt)
    p.add_argument('--dataset', type=str, default=None)
    p.add_argument('--split_idx', type=int, default=None)
    p.add_argument('--fusion_type', type=str, default=None)
    p.add_argument('--diffusion', type=str2bool, default=None)
    p.add_argument('--diffusion_steps', type=int, default=None)

    # figure knobs
    p.add_argument('--n_high', type=int, default=5, help='key frames taken at peaks')
    p.add_argument('--n_low', type=int, default=2, help='key frames taken at valleys')
    p.add_argument('--min_sep_frac', type=float, default=0.07,
                   help='minimum spacing between key frames, as a fraction of the video')
    p.add_argument('--edge_frac', type=float, default=0.02,
                   help='skip this fraction at each end when picking key frames '
                        '(intro/outro cards are usually black); 0 disables')
    p.add_argument('--normalize', type=str2bool, default=True,
                   help='min-max normalise both score curves to [0,1] for display')
    p.add_argument('--smooth', type=int, default=1, help='moving-average window (1 = off)')
    p.add_argument('--fig_width', type=float, default=13.5)
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--pdf', type=str2bool, default=True)
    p.add_argument('--title_metrics', type=str2bool, default=True,
                   help='append F1 / tau / rho to the figure title (turn off for camera-ready)')
    p.add_argument('--save_npz', type=str2bool, default=True)
    args = p.parse_args()

    cfg_d = dict(CONFIG_DEFAULTS)
    cfg_d.update(read_train_config(args.run_dir))
    for key in ('dataset', 'split_idx', 'fusion_type', 'diffusion', 'diffusion_steps'):
        v = getattr(args, key)
        if v is not None:
            cfg_d[key] = v
    cfg_d.setdefault('exp_name', 'exp')
    cfg_d.setdefault('tag', 'tag')
    cfg = Cfg(cfg_d)

    ckpt = find_checkpoint(args.run_dir, args.weights)
    out_dir = args.out_dir or os.path.join(args.run_dir, 'qualitative')
    os.makedirs(out_dir, exist_ok=True)
    args.captions = json.load(open(args.captions_json)) if args.captions_json else None

    print('checkpoint  :', ckpt)
    print('dataset     : {} split {} | fusion {} | diffusion {}'.format(
        cfg.dataset, cfg.split_idx, cfg.fusion_type, cfg.diffusion))
    print('output      :', out_dir)

    if 'summe' in cfg.dataset:
        from utils.summe_dataset import SumMeLLaMADataset as DS
        audio_path = cfg.audio_path or 'audio_features/summe_pann_7.h5'
        visual_path = cfg.visual_path or 'clip_features/clip_summe_7.h5'
        fps_default = 30.0
    else:
        from utils.tvsum_dataset import TVSumLLaMADataset as DS
        audio_path = cfg.audio_path or 'audio_features/tvsum_pann_7.h5'
        visual_path = cfg.visual_path or 'clip_features/clip_tvsum_7.h5'
        fps_default = 30.0

    dataset = DS(mode=args.mode, split_idx=cfg.split_idx, llama_embedding=cfg.pt_path,
                 audio_path=audio_path, visual_path=visual_path, tr_val_ts=cfg.tr_val_ts)
    video_dir = args.videos_dir or DEFAULT_VIDEO_DIRS.get(
        'summe' if 'summe' in cfg.dataset else 'tvsum')
    name_map = build_name_map(cfg.dataset, args.name_map)

    from networks.model import LLMVS
    model = LLMVS.load_from_checkpoint(ckpt, config=cfg, map_location='cpu')
    model.to(cfg.device)
    model.eval()

    rows = []
    for i in range(len(dataset)):
        d = dataset[i]
        video_name = d['video_name']

        x_text = torch.cat((d['llama_embedding_userprompt'],
                            d['llama_embedding_generation']), dim=1).to(cfg.device).float()
        x_audio = d['audio_features'].to(cfg.device).float()
        x_visual = d['visual_features'].to(cfg.device).float()

        with torch.no_grad():
            score = model(x_text, x_audio, x_visual, mask=None).squeeze(1).clamp(0.0, 1.0)
        w = model._last_w.detach().float().cpu().numpy()
        if w.ndim == 1:                       # global fusion -> broadcast to all frames
            w = np.tile(w[None, :], (score.shape[0], 1))

        gt = d['gtscore'].float().cpu().numpy()
        picks = d['picks'].cpu().numpy()
        score_np = score.detach().cpu().numpy()

        machine_summary = generate_summary(score.cpu(), d['change_points'],
                                           [d['n_frames']],
                                           d['n_frame_per_seg'].tolist(), d['picks'])
        kTau, sRho, f1 = evaluate_summary(machine_summary, d['gt_summary'], video_name,
                                          score.cpu(), eval_data=cfg.dataset)

        keyframes = select_keyframes(score_np, args.n_high, args.n_low,
                                     args.min_sep_frac, args.edge_frac)
        video_path = resolve_video_path(video_dir, d.get('video_filename', ''),
                                        video_name, name_map)
        if video_path is None:
            print('[warn] no video file for {} under {}'.format(video_name, video_dir))
        frames, fps = grab_frames(video_path, [picks[idx] for idx, _ in keyframes],
                                  default_fps=fps_default)

        pretty = os.path.splitext(os.path.basename(video_path))[0] if video_path else video_name
        title = '"{}"'.format(pretty.replace('_', ' '))
        if args.title_metrics:
            title += ('   |   F1 = {:.3f}   $\\tau$ = {:.3f}   $\\rho$ = {:.3f}'
                      .format(f1, kTau, sRho))
        out_base = os.path.join(out_dir, '{}_{}'.format(video_name, pretty))

        make_figure(out_base, video_name, title, score_np, gt, w, picks, fps, frames,
                    keyframes, args)

        if args.save_npz:
            np.savez(out_base + '.npz', score=score_np, gtscore=gt, w=w, picks=picks,
                     keyframes=np.array([idx for idx, _ in keyframes]),
                     f1=f1, kTau=kTau, sRho=sRho)

        rows.append(dict(video=video_name, file=pretty, f1=float(f1), kTau=float(kTau),
                         sRho=float(sRho), n_steps=len(score_np),
                         w_text=float(w[:, 0].mean()), w_audio=float(w[:, 1].mean()),
                         w_visual=float(w[:, 2].mean())))
        print('  {:<10s} {:<28s} F1={:.3f}  tau={:.3f}  rho={:.3f}'.format(
            video_name, pretty, f1, kTau, sRho))

    csv_path = os.path.join(out_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    best = max(rows, key=lambda r: r['kTau'])
    print('\nwrote {} figures to {}'.format(len(rows), out_dir))
    print('metrics csv : {}'.format(csv_path))
    print('best tau    : {} ({}) tau={:.3f}'.format(best['video'], best['file'], best['kTau']))


if __name__ == '__main__':
    main()
