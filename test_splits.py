import os
import argparse
import glob
import numpy as np
import wandb
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything

from utils.configs import Config, str2bool
from networks.model import LLMVS
from visualize import make_comparison_plot

parser = argparse.ArgumentParser("check llama score")
parser.add_argument('--dataset', type=str, default='summe')
parser.add_argument('--weights', default='rho', type=str, help="'rho' or 'tau', selects which checkpoint to test")
parser.add_argument('--exp_name', type=str, default='exp_1', help='experiment name used at train time')
parser.add_argument('--seed', type=int, default=1112, help='seed used at train time (selects Summaries/<exp>/seed_<seed>/...)')
parser.add_argument('--result_dir', default='Summaries/', type=str)
parser.add_argument('--pt_path', type=str, default='llama_emb/summe_sum/')
parser.add_argument('--audio_path', type=str, default=None)
parser.add_argument('--visual_path', type=str, default=None)
parser.add_argument('--audio_dim', type=int, default=2048)
parser.add_argument('--visual_dim', type=int, default=768)
parser.add_argument('--reduced_dim', type=int, default=2048)
parser.add_argument('--num_heads', type=int, default=2)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--model', type=str, default='llmvs')
parser.add_argument('--fusion_type', type=str, default='global_weight', choices=['global_weight', 'global_rl', 'local_rl', 'local_weight'], help='the type of fusion used by the checkpoints being tested')
parser.add_argument('--rl_weight', type=float, default=0.1, help='weight of the REINFORCE policy loss when fusion_type=global_rl/local_rl')
parser.add_argument('--baseline_momentum', type=float, default=0.9, help='EMA momentum for the REINFORCE reward baseline')
parser.add_argument('--weight_gen_dim', type=int, default=256, help='per-modality compression dim before the per-frame weight generator (fusion_type=local_weight/local_rl)')
parser.add_argument('--wandb', type=str2bool, default=True, help='log eval results to wandb')
parser.add_argument('--wandb_ckpt', type=str2bool, default=False, help='also upload checkpoints as a wandb artifact (large, counts against storage quota)')
parser.add_argument('--wandb_entity', type=str, default='dogann19-istanbul-technical-university')
parser.add_argument('--wandb_project', type=str, default='Video_Summ_3_Modal')

args = parser.parse_args()
seed_everything(args.seed)

if args.wandb:
    wandb_run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, group=f'{args.exp_name}_{args.dataset}', job_type='eval',
                            name=f'{args.exp_name}_seed{args.seed}_{args.dataset}_{args.weights}', config=vars(args))
    if args.wandb_ckpt:
        ckpt_artifact = wandb.Artifact(f'{args.exp_name}-seed{args.seed}-{args.dataset}-{args.weights}-checkpoints', type='model')

ckpt_dirname = 'best_rho_model' if 'rho' in args.weights else 'best_tau_model'

exp_dataset_root = 'Summaries/{}/seed_{}/{}'.format(args.exp_name, args.seed, args.dataset)
tags = sorted(glob.glob('{}/*'.format(exp_dataset_root)))
tags = [t.split('/')[-1] for t in tags if os.path.isdir(t)]

if args.dataset == 'summe':
    from utils.summe_dataset import SumMeLLaMADataset as Dataset, ValBatchCollator
    default_audio_path = 'audio_features/summe_pann_7.h5'
    default_visual_path = 'clip_features/clip_summe_7.h5'
else:
    from utils.tvsum_dataset import TVSumLLaMADataset as Dataset, ValBatchCollator
    default_audio_path = 'audio_features/tvsum_pann_7.h5'
    default_visual_path = 'clip_features/clip_tvsum_7.h5'

audio_path = args.audio_path or default_audio_path
visual_path = args.visual_path or default_visual_path

kTau_scores, sRho_scores, f1_scores = [], [], []
per_split_rows = []

for split_idx, tag in enumerate(tags[:5]):
    ckpt_glob = sorted(glob.glob('{}/{}/{}/epoch=*'.format(exp_dataset_root, tag, ckpt_dirname)))
    if not ckpt_glob:
        print('No checkpoint found for split {} ({}), skipping'.format(split_idx, tag))
        continue
    weights_path = ckpt_glob[0]

    config = Config(model=args.model, dataset=args.dataset, split_idx=split_idx, epochs=1,
                     reduced_dim=args.reduced_dim, num_heads=args.num_heads, num_layers=args.num_layers,
                     tag=tag, seed=args.seed, pt_path=args.pt_path, audio_path=args.audio_path, visual_path=args.visual_path,
                     audio_dim=args.audio_dim, visual_dim=args.visual_dim, exp_name=args.exp_name,
                     weights=weights_path, fusion_type=args.fusion_type, weight_gen_dim=args.weight_gen_dim,
                     rl_weight=args.rl_weight, baseline_momentum=args.baseline_momentum)

    test_dataset = Dataset(mode='test', split_idx=split_idx, llama_embedding=config.pt_path,
                            audio_path=audio_path, visual_path=visual_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8,
                              collate_fn=ValBatchCollator(), pin_memory=True)

    model = LLMVS.load_from_checkpoint(weights_path, config=config)
    model.cuda()
    model.eval()

    trainer = Trainer(gpus=1, precision=16, benchmark=True, deterministic=False, logger=False,
                       progress_bar_refresh_rate=0)
    results = trainer.test(model, test_loader, ckpt_path=weights_path, verbose=False)[0]

    kTau_scores.append(results['val_kTau'])
    sRho_scores.append(results['val_sRho'])
    f1_scores.append(results['val_f1'])
    per_split_rows.append('{}: kTau={:.3f} sRho={:.3f} f1={:.3f}'.format(
        tag, results['val_kTau'], results['val_sRho'], results['val_f1']))
    print(per_split_rows[-1])

    if args.wandb and args.wandb_ckpt:
        ckpt_artifact.add_file(weights_path, name=f'{tag}/{os.path.basename(weights_path)}')

kTau_scores = np.array(kTau_scores)
sRho_scores = np.array(sRho_scores)
f1_scores = np.array(f1_scores)

summary_lines = [
    'kTau: {:.3f} +/- {:.3f}'.format(kTau_scores.mean(), kTau_scores.std()),
    'sRho: {:.3f} +/- {:.3f}'.format(sRho_scores.mean(), sRho_scores.std()),
    'f1:   {:.3f} +/- {:.3f}'.format(f1_scores.mean(), f1_scores.std()),
]
for line in summary_lines:
    print(line)

result_dir = os.path.join(args.result_dir, args.exp_name, 'seed_{}'.format(args.seed))
os.makedirs(result_dir, exist_ok=True)
result_file = os.path.join(result_dir, '{}_{}_results.txt'.format(args.dataset, args.weights))
with open(result_file, 'w') as f:
    f.write('\n'.join(per_split_rows))
    f.write('\n')
    f.write('\n'.join(summary_lines))
    f.write('\n')

if args.wandb:
    table = wandb.Table(columns=['split', 'kTau', 'sRho', 'f1'])
    for tag, k, s, f1 in zip(tags, kTau_scores, sRho_scores, f1_scores):
        table.add_data(tag, float(k), float(s), float(f1))

    wandb_run.log({
        'per_split_results': table,
        'kTau_mean': kTau_scores.mean(), 'kTau_std': kTau_scores.std(),
        'sRho_mean': sRho_scores.mean(), 'sRho_std': sRho_scores.std(),
        'f1_mean': f1_scores.mean(), 'f1_std': f1_scores.std(),
    })

    results_artifact = wandb.Artifact(f'{args.exp_name}-seed{args.seed}-{args.dataset}-{args.weights}-results', type='results')
    results_artifact.add_file(result_file)
    wandb_run.log_artifact(results_artifact)
    if args.wandb_ckpt:
        wandb_run.log_artifact(ckpt_artifact)

    plot_path = os.path.join(result_dir, '{}_{}_comparison.png'.format(args.dataset, args.weights))
    if make_comparison_plot(exp_dataset_root, plot_path) is not None:
        wandb_run.log({'comparison': wandb.Image(plot_path)})
        plot_artifact = wandb.Artifact(f'{args.exp_name}-seed{args.seed}-{args.dataset}-{args.weights}-comparison', type='visualization')
        plot_artifact.add_file(plot_path)
        wandb_run.log_artifact(plot_artifact)

    wandb_run.finish()
