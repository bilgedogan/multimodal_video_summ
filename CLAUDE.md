# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multimodal extension of LLMVS (CVPR 2025) — video summarization using LLaMA text embeddings, PANN audio features, and CLIP visual features fused through a Transformer encoder. Benchmarks on SumMe and TVSum with 5-fold cross-validation. Active research branch: `multimodal`.

## Environment

```bash
conda activate llmvs   # Python 3.8, PyTorch 1.13.1+cu117, pytorch-lightning 1.5.10
```

## Commands

### Train one split
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset summe --split_idx 0 --tag summe_split0 \
  --exp_name exp_1 --fusion_type global_weight \
  --lr 0.000119 --epochs 200 \
  --reduced_dim 2048 --num_heads 2 --num_layers 3 \
  --pt_path llama_emb/summe_sum/
```
Saved to `Summaries/<exp_name>/seed_<seed>/<dataset>/<tag>/`.

### Run a predefined experiment (all 5 splits)
```bash
bash configs/exp_1_summe.sh   # global_weight
bash configs/exp_2_summe.sh   # global_rl
bash configs/exp_3_summe.sh   # local_weight
bash configs/exp_4_summe.sh   # local_rl
```

### Seed sweep (5 seeds × 5 splits, then eval)
```bash
bash train_seed_sweep.sh <dataset> <exp_name> <fusion_type>
# e.g.: bash train_seed_sweep.sh summe exp_1 global_weight
```

### Evaluate all splits for an experiment (preferred)
```bash
# Reports mean ± std across splits; logs to WandB
python test_splits.py --dataset summe --weights rho \
  --exp_name exp_1 --seed 1112 --fusion_type global_weight \
  --pt_path llama_emb/summe_sum/
```
Convenience wrappers: `test_summe.sh`, `test_tvsum.sh`.

### Evaluate a single checkpoint
```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --dataset summe --split_idx 0 --tag summe_split0 \
  --weights Summaries/exp_1/seed_1112/summe/summe_split0/best_rho_model/epoch=XX-val_sRho=X.XXX.ckpt \
  --fusion_type global_weight --pt_path llama_emb/summe_sum/
```

### Visualize training runs
```bash
python visualize.py --logdir Summaries --out compare.png
python visualize.py --logdir Summaries/exp_1 --out compare.png
```
Reads TensorBoard event files; optionally uploads to WandB.

## Architecture

```
Per-frame features (text, audio, visual)
  → linear projection to reduced_dim (default 2048) + LayerNorm
  → fusion (one of four strategies, see below)
  → TransformerEncoder (num_heads=2, num_layers=3)
  → MLP head → sigmoid → frame importance score ∈ [0,1]
```

**Text input**: two LLaMA embedding streams (`user_prompt_pool.h5` + `gen_pool.h5`) concatenated on the token dim, then max-pooled per frame (5120 → reduced_dim).  
**Audio input**: PANN features (2048-dim), linear → reduced_dim.  
**Visual input**: CLIP features (768-dim), linear → reduced_dim.

### Fusion types (`--fusion_type`)
| Value | Mechanism |
|---|---|
| `global_weight` | Single learned softmax weight vector over 3 modalities (default) |
| `global_rl` | Dirichlet distribution over modalities; weights sampled via REINFORCE |
| `local_weight` | Per-frame softmax weights predicted by a small MLP |
| `local_rl` | Per-frame Dirichlet; weights sampled via REINFORCE per frame |

### Deterministic fusion details

**`global_weight`** — one weight vector, input-independent:
```
modality_weights  (shape 3)  = nn.Parameter(ones(3))   # learned constants
w                 (shape 3)  = softmax(modality_weights)
x = w[0]*x_text + w[1]*x_audio + w[2]*x_visual         # same w for every frame
```
Three scalars optimised by backprop through MSELoss. No frame-content awareness.

**`local_weight`** — per-frame weights predicted from frame content:
```
wt (shape N×weight_gen_dim) = relu(wg_text(x_text))    # Linear(reduced_dim, weight_gen_dim)
wa                           = relu(wg_audio(x_audio))
wv                           = relu(wg_visual(x_visual))
raw (shape N×3)              = frame_weight_gen(cat(wt, wa, wv, dim=-1))
w   (shape N×3)              = softmax(raw, dim=-1)
x = w[:,0:1]*x_text + w[:,1:2]*x_audio + w[:,2:3]*x_visual
```
Weights are a function of each frame's projected features → modality importance varies frame by frame. Adds `wg_text`, `wg_audio`, `wg_visual` (each `Linear(reduced_dim, weight_gen_dim)`) and `frame_weight_gen` (`Linear(weight_gen_dim*3, 3)`). `--weight_gen_dim` controls bottleneck size (default 256).

**Key axis of difference across all four types:**

| | Granularity | Input-dependent | Stochastic (training) |
|---|---|---|---|
| `global_weight` | whole video | no | no |
| `global_rl` | whole video | no | yes (Dirichlet sample) |
| `local_weight` | per frame | yes | no |
| `local_rl` | per frame | yes | yes (Dirichlet sample) |

### REINFORCE details

Both RL variants share the same reward signal and baseline update in `training_step`. The key difference is the **granularity of the action** (global vs. per-frame).

**Shared reward / baseline / loss** (`networks/model.py:156–167`):
```
reward  = Kendall's τ(predicted_scores, gt_scores)   # scalar, per video
baseline ← momentum * baseline + (1 - momentum) * reward  # EMA, --baseline_momentum=0.9
advantage = reward - baseline
policy_loss = -mean(log_prob) * advantage.detach()
total_loss = mse_loss + rl_weight * policy_loss        # --rl_weight=0.1
```

**`global_rl`** — one Dirichlet over the whole video:
```
alpha (shape 3)  = softplus(modality_alpha_raw) + 1   # learnable nn.Parameter
w     (shape 3)  = Dirichlet(alpha).rsample()          # single draw, training only
x = w[0]*x_text + w[1]*x_audio + w[2]*x_visual        # same w for every frame
log_prob         = Dirichlet(alpha).log_prob(w)        # scalar
```
At inference: `w = alpha / alpha.sum()` (deterministic Dirichlet mean).

**`local_rl`** — one Dirichlet **per frame**:
```
raw   (shape N×3) = frame_weight_gen(cat(relu(wg_text(x_text)),
                                         relu(wg_audio(x_audio)),
                                         relu(wg_visual(x_visual)), dim=-1))
alpha (shape N×3) = softplus(raw) + 1
w     (shape N×3) = Dirichlet(alpha).rsample()          # one draw per frame
x = w[:,0:1]*x_text + w[:,1:2]*x_audio + w[:,2:3]*x_visual
log_prob (shape N) = Dirichlet(alpha).log_prob(w)
# in training_step: log_prob.mean() used (average over frames) with same scalar reward
```
At inference: `w = alpha / alpha.sum(dim=-1, keepdim=True)` per frame.

The reward is always a single scalar τ per video regardless of fusion type; `local_rl` averages per-frame log-probs before multiplying by advantage.

### Diffusion denoising of the fused vector (`--diffusion True`)

`networks/diffusion.py:DiffusionDenoiser` — a DDPM ε-predictor (MLP: `Linear(reduced_dim, reduced_dim//2)` + sinusoidal timestep embedding → SiLU MLP → `Linear(reduced_dim//2, reduced_dim)`) applied to the fused vector `x` (N × reduced_dim), between fusion and the Transformer encoder. Linear β schedule 1e-4 → 0.02 over `--diffusion_steps` (default 20).

```
x_fused                                  # after modality weighting
x_den = diff_net.denoise(x_fused)        # DDIM (eta=0) reverse chain T-1 → 0, no_grad,
                                         # fused vector treated as x_T (mild noise level)
x = x_fused + (x_den - x_fused).detach() # value = x_den; identity gradient to fusion
```

**Gradient isolation** (the point of the `.detach()`): MSE / diversity / REINFORCE losses never reach `diff_net`, and the diffusion loss never reaches fusion or the head.

```
t     ~ U{0..T-1} per frame
x_t   = sqrt(acp[t]) * x_fused.detach() + sqrt(1-acp[t]) * noise
diff_loss = MSE(diff_net(x_t, t), noise)
total_loss = task_loss + diffusion_weight * diff_loss    # --diffusion_weight, default 0.1
```

`--diffusion` and `--diffusion_steps` must match at eval (`test.py`, `test_splits.py`) or the checkpoint won't load. Predefined runs: `configs/exp_2_summe_diff*.sh` (global_rl), `configs/exp_4_summe_diff*.sh` (local_rl). Logged as `diffusion_loss`.

### Loss & metrics
- Training: MSELoss on frame scores (+ optional REINFORCE policy loss, + optional diffusion ε-loss)
- Validation/test: F1 (knapsack-based binary summary), Kendall's τ, Spearman's ρ
- Checkpoints saved for best `val_sRho` and best `val_kTau` separately

## Data Paths

| Data | Default path |
|---|---|
| LLaMA embeddings | `llama_emb/summe_sum/` or `llama_emb/tvsum_sum/` |
| Audio (training default) | `audio_features/<dataset>_pann_7.h5` |
| Visual (training default) | `clip_features/clip_<dataset>_7.h5` |
| SumMe video features | `SumMe/eccv16_dataset_summe_google_pool5.h5` |
| TVSum video features | `TVSum/eccv16_dataset_tvsum_google_pool5.h5` |
| Split definitions | `dataset/summe_splits.json`, `dataset/tvsum_splits.json` |

### Train/val/test splits (`--tr_val_ts`)

Default (`False`): train on `train_keys`, validate on `test_keys` (original LLMVS protocol).

With `--tr_val_ts True`: `train_keys` is cut in order — first 80% stay train, last 20% become a held-out val set used for validation and checkpoint selection. `test_keys` is untouched, so `test.py` / `test_splits.py` still report on the same test videos.

| Dataset | train : val : test |
|---|---|
| SumMe | 16 : 4 : 5 |
| TVSum | 32 : 8 : 10 |

The cut is a positional slice of the JSON key order — no shuffling, no RNG, so it is identical across seeds and runs. `mode='val'` on the dataset classes raises unless `tr_val_ts=True`.

Alternative audio/visual files (`_pann_5.h5`, `_whisper.h5`, non-7 clip files) exist but are not the training default. Override with `--audio_path` / `--visual_path`.

## WandB

Project: `Video_Summ_3_Modal`, entity: `dogann19-istanbul-technical-university`.  
`test_splits.py` and `visualize.py` log to WandB by default (`--wandb True`). Disable with `--wandb False`.
