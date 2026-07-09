import torch
from torch import nn
from einops import repeat

import pytorch_lightning as pl
from utils.evaluation_metrics import evaluate_summary
from utils.generate_summary import generate_summary
from pytorch_lightning import seed_everything
import pdb
import numpy as np
import torch.nn.functional as F
from scipy import stats
from matplotlib import pyplot as plt
seed_everything(1112)

class LLMVS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.d_max_pooling = nn.AdaptiveMaxPool1d(self.config.reduced_dim)
        self.d_linear1 = nn.Linear(5120, self.config.reduced_dim)
        self.d_linear1_norm = nn.LayerNorm(self.config.reduced_dim)
        self.c_max_pooling = nn.AdaptiveMaxPool1d(1)

        self.a_linear1 = nn.Linear(self.config.audio_dim, self.config.reduced_dim)
        self.a_linear1_norm = nn.LayerNorm(self.config.reduced_dim)
        self.v_linear1 = nn.Linear(self.config.visual_dim, self.config.reduced_dim)
        self.v_linear1_norm = nn.LayerNorm(self.config.reduced_dim)

        if self.config.fusion_type == 'global_rl':
            self.modality_alpha_raw = nn.Parameter(torch.zeros(3))
            self.register_buffer('reward_baseline', torch.zeros(1))
        elif self.config.fusion_type in ('local_weight', 'local_rl'):
            self.wg_text = nn.Linear(self.config.reduced_dim, self.config.weight_gen_dim)
            self.wg_audio = nn.Linear(self.config.reduced_dim, self.config.weight_gen_dim)
            self.wg_visual = nn.Linear(self.config.reduced_dim, self.config.weight_gen_dim)
            self.frame_weight_gen = nn.Linear(self.config.weight_gen_dim * 3, 3)
            if self.config.fusion_type == 'local_rl':
                self.register_buffer('reward_baseline', torch.zeros(1))
        else:
            self.modality_weights = nn.Parameter(torch.ones(3))

        encoder_layer_agg = nn.TransformerEncoderLayer(d_model = self.config.reduced_dim, nhead = self.config.num_heads, batch_first = True)
        self.transformer_encoder_agg = nn.TransformerEncoder(encoder_layer_agg, num_layers=self.config.num_layers)

        self.mlp_head = torch.nn.Sequential(
            nn.Linear(self.config.reduced_dim, self.config.reduced_dim//2),
            nn.LayerNorm(self.config.reduced_dim // 2), 
            nn.ReLU(),
            nn.Linear(self.config.reduced_dim//2, self.config.reduced_dim//4),
            nn.ReLU(),
            nn.Linear(self.config.reduced_dim//4, self.config.reduced_dim//8),
            nn.LayerNorm(self.config.reduced_dim // 8),
            nn.ReLU(),
            nn.Linear(self.config.reduced_dim//8, self.config.reduced_dim//16),
            nn.ReLU(),
            nn.Linear(self.config.reduced_dim//16, 1),
            nn.Sigmoid()
        )
                
        self.criterion = nn.MSELoss()

        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_text, x_audio, x_visual, mask):

        x_text = self.c_max_pooling(x_text.permute(0,2,1)).squeeze(2)
        x_text = self.d_linear1(x_text)
        x_text = self.d_linear1_norm(x_text)

        x_audio = self.a_linear1(x_audio)
        x_audio = self.a_linear1_norm(x_audio)

        x_visual = self.v_linear1(x_visual)
        x_visual = self.v_linear1_norm(x_visual)

        if self.config.fusion_type == 'global_rl':
            alpha = F.softplus(self.modality_alpha_raw) + 1e-3
            alpha = alpha.clamp(0.1, 20)
            dist = torch.distributions.Dirichlet(alpha)
            if self.training:
                w = dist.rsample()
                self._fusion_log_prob = dist.log_prob(w)
            else:
                w = alpha / alpha.sum()
                self._fusion_log_prob = None
            x = w[0] * x_text + w[1] * x_audio + w[2] * x_visual
        elif self.config.fusion_type in ('local_weight', 'local_rl'):
            wt = F.relu(self.wg_text(x_text))
            wa = F.relu(self.wg_audio(x_audio))
            wv = F.relu(self.wg_visual(x_visual))
            raw = self.frame_weight_gen(torch.cat((wt, wa, wv), dim=-1))

            if self.config.fusion_type == 'local_rl':
                alpha = F.softplus(raw) + 1
                dist = torch.distributions.Dirichlet(alpha)
                if self.training:
                    w = dist.rsample()
                    self._fusion_log_prob = dist.log_prob(w)
                else:
                    w = alpha / alpha.sum(dim=-1, keepdim=True)
                    self._fusion_log_prob = None
            else:
                w = torch.softmax(raw, dim=-1)

            x = w[:, 0:1] * x_text + w[:, 1:2] * x_audio + w[:, 2:3] * x_visual
        else:
            w = torch.softmax(self.modality_weights, dim=0)
            x = w[0] * x_text + w[1] * x_audio + w[2] * x_visual

        self._last_w = w.detach()
        x = x.unsqueeze(0)

        x = self.transformer_encoder_agg(x)
        x = self.mlp_head(x.squeeze(0))

        return  x

    def _log_fusion_weights(self, stage):
        w = self._last_w
        if w.dim() == 1:
            w_text, w_audio, w_visual = w[0], w[1], w[2]
        else:
            w_mean = w.mean(dim=0)
            w_text, w_audio, w_visual = w_mean[0], w_mean[1], w_mean[2]
            if stage == 'val' and self.logger is not None:
                self.logger.experiment.add_histogram(f'{stage}_w_text_hist', w[:, 0], self.global_step)
                self.logger.experiment.add_histogram(f'{stage}_w_audio_hist', w[:, 1], self.global_step)
                self.logger.experiment.add_histogram(f'{stage}_w_visual_hist', w[:, 2], self.global_step)

        self.log(f'{stage}_w_text', w_text, on_step=(stage == 'train'), on_epoch=True, batch_size=1)
        self.log(f'{stage}_w_audio', w_audio, on_step=(stage == 'train'), on_epoch=True, batch_size=1)
        self.log(f'{stage}_w_visual', w_visual, on_step=(stage == 'train'), on_epoch=True, batch_size=1)

    def training_step(self, train_batch, batch_idx):
        x1 = train_batch['llama_embedding_userprompt'].squeeze(0)
        x2 = train_batch['llama_embedding_generation'].squeeze(0)

        x_text = torch.cat((x1, x2), dim=1)
        x_audio = train_batch['audio_features'].squeeze(0)
        x_visual = train_batch['visual_features'].squeeze(0)

        y = train_batch['gtscore']
        mask = train_batch['mask']

        score = self.forward(x_text, x_audio, x_visual, mask=mask).squeeze(1).unsqueeze(0)
        del x_text, x_audio, x_visual, mask
        score = score.clamp(0.0, 1.0)

        loss = self.criterion(score, y).mean()

        if self.config.fusion_type in ('global_rl', 'local_rl'):
            pred_rank = stats.rankdata(-score.detach().squeeze().cpu().numpy())
            gt_rank = stats.rankdata(-y.detach().squeeze().cpu().numpy())
            tau = stats.kendalltau(pred_rank, gt_rank)[0]
            reward = torch.tensor(0.0 if np.isnan(tau) else tau, device=score.device)

            self.reward_baseline = self.config.baseline_momentum * self.reward_baseline + (1 - self.config.baseline_momentum) * reward
            advantage = reward - self.reward_baseline

            log_prob = self._fusion_log_prob.mean() if self._fusion_log_prob.dim() > 0 else self._fusion_log_prob
            policy_loss = -log_prob * advantage.detach()
            loss = loss + self.config.rl_weight * policy_loss

            self.log('reward_tau', reward, on_step=True, on_epoch=True, batch_size=1)
            self.log('policy_loss', policy_loss, on_step=True, on_epoch=True, batch_size=1)

        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size = 1)
        self._log_fusion_weights('train')

        torch.cuda.empty_cache()
        return loss

    def validation_step(self, val_batch, batch_idx):
        x1 = val_batch['llama_embedding_userprompt'].squeeze(0)
        x2 = val_batch['llama_embedding_generation'].squeeze(0)

        x_text = torch.cat((x1, x2), dim=1)
        x_audio = val_batch['audio_features'].squeeze(0)
        x_visual = val_batch['visual_features'].squeeze(0)
        mask = val_batch['mask']

        score = self.forward(x_text, x_audio, x_visual, mask=mask).squeeze(1).unsqueeze(0)
        del x_text, x_audio, x_visual, mask
        score = score.clamp(0.0, 1.0)
        self._log_fusion_weights('val')

        score = score.squeeze()
        gt_summary = val_batch['gt_summary'][0]
        cps = val_batch['change_points'][0]
        n_frames = val_batch['n_frames']
        nfps = val_batch['n_frame_per_seg'][0].tolist()
        video_name = val_batch['video_name'][0]
 
        picks = val_batch['picks'][0]
        machine_summary = generate_summary(score, cps, n_frames, nfps, picks)

        kTau, sRho, f1 = evaluate_summary(machine_summary, gt_summary, video_name, score, eval_data=self.config.dataset)

        return kTau, sRho, f1

    def validation_epoch_end(self, outs):
        outs = torch.tensor(outs)

        kTau = outs[:,0].mean()
        sRho = outs[:,1].mean()
        f1 = outs[:,2].mean()

        self.log('val_kTau', kTau, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_sRho', sRho, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True)

        torch.cuda.empty_cache()

    def test_step(self, val_batch, batch_idx):
        x1 = val_batch['llama_embedding_userprompt'].squeeze(0)
        x2 = val_batch['llama_embedding_generation'].squeeze(0)

        x_text = torch.cat((x1, x2), dim=1)
        x_audio = val_batch['audio_features'].squeeze(0)
        x_visual = val_batch['visual_features'].squeeze(0)
        y = val_batch['gtscore'].squeeze(0)
        mask = val_batch['mask']

        score = self.forward(x_text, x_audio, x_visual, mask=mask).squeeze(1).unsqueeze(0)
        del x_text, x_audio, x_visual, mask
        score = score.clamp(0.0, 1.0)

        score = score.squeeze()

        gt_summary = val_batch['gt_summary'][0]
        cps = val_batch['change_points'][0]
        n_frames = val_batch['n_frames']
        nfps = val_batch['n_frame_per_seg'][0].tolist()
        video_name = val_batch['video_name'][0]

        picks = val_batch['picks'][0]
        machine_summary = generate_summary(score, cps, n_frames, nfps, picks)

        kTau, sRho, f1 = evaluate_summary(machine_summary, gt_summary, video_name, score, eval_data=self.config.dataset)


        return kTau, sRho, f1

    def test_epoch_end(self, outs):
        outs = torch.tensor(outs)

        kTau = outs[:,0].mean()
        sRho = outs[:,1].mean()
        f1 = outs[:,2].mean()

        self.log('val_kTau', kTau, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_sRho', sRho, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):        
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6),
            'interval': 'epoch',
            'frequency': 1,
        }
        return [optimizer], [lr_scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)