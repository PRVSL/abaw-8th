"""
Author: PRVSL
https://pr.ai.vn
"""
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torchmetrics import F1Score, PearsonCorrCoef, MeanSquaredError

import torch
from torch.nn import functional as F
from torch import nn
import timm
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.utils.model import freeze, unfreeze, freeze_batch_norm_2d

from utils import SigmoidFocalLoss, CCCLoss, CEFocalLoss, ConCorrCoef

from functools import partial
from einops.layers.torch import Rearrange, Reduce



class ABAWModel(pl.LightningModule):
    def __init__(self, task, seq_len, learning_rate=2e-4, focal_alpha=0.75, focal_gamma=2., label_smoothing=0.1,
                 wd=2e-5, threshold=0.5, batch_size=32, n_mixer=3):
        super().__init__()
        self.save_hyperparameters()

        self.task = task
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.threshold = threshold
        self.get_metrics_criterion(focal_alpha, focal_gamma, label_smoothing)

        self.learning_rate = learning_rate
        self.wd = wd
        self.n_mixer = n_mixer

        self.backbone = timm.create_model('mobilenetv4_conv_small',  # 'hiera_tiny_224.mae',
                                          pretrained=True,
                                          features_only=True,
                                          pretrained_cfg={
                                              'file': './pretrained-affectnet/model_best.pth.tar'}
                                          )
        dim = 256
        depth = 1
        output_dim = 256
        if self.n_mixer >= 3:
            self.mid_mixer = MLPMixer3D(image_size=(28, 28), time_size=self.seq_len,
                                        time_patch_size=4, channels=64, patch_size=4,
                                        dim=dim, depth=depth, output_dim=output_dim)
        if self.n_mixer >= 2:
            self.low_mixer = MLPMixer3D(image_size=(14, 14), time_size=self.seq_len,
                                        time_patch_size=4, channels=96, patch_size=2,
                                        dim=dim, depth=depth, output_dim=output_dim)
        if self.n_mixer >= 1:
            self.last_mixer = MLPMixer3D(image_size=(7, 7), time_size=self.seq_len,
                                         time_patch_size=4, channels=960, patch_size=1,
                                         dim=dim, depth=depth, output_dim=output_dim)
        self.dropout = nn.Dropout(p=0.3)

        submodules = [n for n, _ in self.backbone.named_children()]
        freeze(self.backbone, submodules[:-1])
        self.n_features = output_dim * min(3, self.n_mixer)  # 440
        self.classifier = nn.Linear(self.n_features, self.num_outputs)

    def forward(self, x):
        """
        :param x: 5-D vector, batch_size x seq_len x n_channels x H x W

        """
        num_seq = x.shape[0]
        x = torch.reshape(x, (num_seq * self.seq_len,) + x.shape[2:])

        feats = self.backbone(x)  # batch size * seq x num_feat

        feat_list = []
        if self.n_mixer >= 3:
            feat_mid = torch.reshape(feats[-3], (num_seq, self.seq_len, -1, 14*2, 14*2))
            feat_mid = self.mid_mixer(feat_mid)
            feat_list.append(feat_mid)

        if self.n_mixer >= 2:
            feat_low = torch.reshape(feats[-2], (num_seq, self.seq_len, -1, 7*2, 7*2))
            feat_low = self.low_mixer(feat_low)
            feat_list.append(feat_low)

        feat_last = torch.reshape(feats[-1], (num_seq, self.seq_len, -1, 7, 7))
        feat_last = self.last_mixer(feat_last)
        feat_list.append(feat_last)

        if len(feat_list) > 1:
            out = torch.cat(feat_list, dim=-1)
        else:
            out = feat_list[0]

        out = self.dropout(out)
        out = self.classifier(out)
        out = torch.unsqueeze(out, dim=1)
        out = out.expand(-1, self.seq_len, -1)
        if self.task == 'VA':
            out = F.tanh(out)
        return out

    def training_step(self, batch):
        x = batch['image']
        y = batch[self.task]

        logits = self(x)
        loss = self.loss_func(logits, y)

        self.update_metric(logits, y, is_train=True)
        self.log('train_metric', self.train_metric, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch[self.task]

        logits = self(x)
        loss = self.loss_func(logits, y)

        self.update_metric(logits, y, is_train=False)
        self.log_dict({'val_metric': self.val_metric, 'val_loss': loss}, on_step=False, on_epoch=True, prog_bar=True,
                      batch_size=self.batch_size)

    def predict_step(self, batch, batch_idx):
        x = batch['image']
        logits = self(x)
        if self.task == 'VA':
            out = logits
        elif self.task == 'AU':
            out = F.sigmoid(logits)
        elif self.task == 'EXPR':
            out = F.softmax(logits)
        else:
            raise NotImplementedError

        # out = torch.reshape(out, (-1, self.num_outputs))
        return out, batch['video_id']

    def configure_optimizers(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = create_optimizer_v2(model_parameters, opt='adamw', lr=self.learning_rate, weight_decay=self.wd)
        tmax = 500 * 4
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=tmax, T_mult=1, eta_min=1e-6, )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def get_metrics_criterion(self, focal_alpha, focal_gamma, label_smoothing):
        if self.task == 'VA':
            # Regression
            self.num_outputs = 2
            self.loss_func = partial(CCCLoss)

            self.train_metric = ConCorrCoef(num_classes=self.num_outputs)
            self.val_metric = ConCorrCoef(num_classes=self.num_outputs)

        elif self.task == 'EXPR':
            # Classification
            self.num_outputs = 8
            self.label_smoothing = label_smoothing

            self.loss_func = partial(CEFocalLoss, num_classes=self.num_outputs,
                                     label_smoothing=label_smoothing,
                                     alpha=focal_alpha, gamma=focal_gamma)

            self.train_metric = F1Score(threshold=self.threshold, num_classes=self.num_outputs, average='macro',
                                        task='multiclass')
            self.val_metric = F1Score(threshold=self.threshold, num_classes=self.num_outputs, average='macro',
                                      task='multiclass')

        elif self.task == 'AU':
            # Multi-label classification
            self.num_outputs = 12

            self.loss_func = partial(SigmoidFocalLoss, num_classes=self.num_outputs,
                                     alpha=focal_alpha, gamma=focal_gamma)

            self.train_metric = F1Score(threshold=self.threshold, num_classes=self.num_outputs,
                                        num_labels=self.num_outputs, average='macro', task='multilabel')
            self.val_metric = F1Score(threshold=self.threshold, num_classes=self.num_outputs,
                                      num_labels=self.num_outputs, average='macro', task='multilabel')

    def update_metric(self, out, y, is_train=True):
        if self.task == 'EXPR':
            y = torch.reshape(y, (-1,))
            # out = F.softmax(out, dim=1)
        elif self.task == 'AU':
            out = torch.sigmoid(out)
            y = torch.reshape(y, (-1, self.num_outputs))

        elif self.task == 'VA':
            y = torch.reshape(y, (-1, self.num_outputs))

        out = torch.reshape(out, (-1, self.num_outputs))

        if is_train:
            self.train_metric(out, y)
        else:
            self.val_metric(out, y)


"""
MLP Mixer by https://github.com/lucidrains/mlp-mixer-pytorch
"""
pair = lambda x: x if isinstance(x, tuple) else (x, x)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


def MLPMixer3D(*, image_size, time_size, channels, patch_size, time_patch_size, dim, depth, output_dim,
               expansion_factor=4, expansion_factor_token=0.5, dropout=0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    assert (time_size % time_patch_size) == 0, 'time dimension must be divisible by time patch size'

    num_patches = (image_h // patch_size) * (image_w // patch_size) * (time_size // time_patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

    return nn.Sequential(
        Rearrange('b t c h w -> b c t h w'),
        Rearrange('b c (t pt) (h p1) (w p2) -> b (h w t) (p1 p2 pt c)', p1=patch_size, p2=patch_size,
                  pt=time_patch_size),
        nn.Linear((time_patch_size * patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, output_dim)
    )
