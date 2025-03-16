"""
Author: Huynh Van Thong
https://pr.ai.vn
"""

import torch
from torch.nn import functional as F

from typing import Tuple
from torch import Tensor
import torchmetrics
from torchmetrics.utilities.checks import _check_same_shape


def CCCLoss(y_hat, y, scale_factor=1., num_classes=2):
    y_hat_fl = torch.reshape(y_hat, (-1, num_classes))
    y_fl = torch.reshape(y, (-1, num_classes))

    yhat_mean = torch.mean(y_hat_fl, dim=0, keepdim=True)
    y_mean = torch.mean(y_fl, dim=0, keepdim=True)

    sxy = torch.mean(torch.mul(y_fl - y_mean, y_hat_fl - yhat_mean), dim=0)
    rhoc = torch.div(2 * sxy,
                     torch.var(y_fl, dim=0) + torch.var(y_hat_fl, dim=0) + torch.square(y_mean - yhat_mean) + 1e-8)

    return 1 - torch.mean(rhoc)


def SigmoidFocalLoss(y_hat, y, scale_factor=1, num_classes=12, pos_weights=None, alpha=0.25, gamma=2):
    y_hat_fl = torch.reshape(y_hat, (-1, num_classes))
    y_fl = torch.reshape(y, (-1, num_classes)) * 1.0
    loss_val = sigmoid_focal_loss(inputs=y_hat_fl, targets=y_fl, alpha=alpha, gamma=gamma, reduction='mean')
    return loss_val


def CEFocalLoss(y_hat, y, scale_factor=1, num_classes=8, label_smoothing=0., class_weights=None, alpha=0.25, gamma=2.,
                distillation_loss=True):
    y_hat_fl = torch.reshape(y_hat, (-1, num_classes))
    y_fl = torch.reshape(y, (-1,))  # Class indices

    ce_loss = F.cross_entropy(y_hat_fl, y_fl, label_smoothing=label_smoothing, reduction='none')
    target_one_hot = F.one_hot(y_fl, num_classes=num_classes)
    p = F.softmax(y_hat_fl, dim=1)

    if distillation_loss:
        target_one_hot_smooth = target_one_hot * (1 - label_smoothing) + label_smoothing / num_classes
        dist_loss = F.kl_div(F.log_softmax(y_hat_fl, dim=1), target_one_hot_smooth, reduction='batchmean')
    else:
        dist_loss = 0.

    p_t = torch.sum(p * target_one_hot, dim=1)  # + (1 - p) * (1 - target_one_hot)
    loss = ce_loss * torch.pow(1 - p_t, gamma)
    if alpha > 0.:
        alpha_t = torch.sum(alpha * target_one_hot, dim=1)
        loss = alpha_t * loss

    if distillation_loss:
        dist_loss_coeff = 0.2
        return loss.mean() * (1 - dist_loss_coeff) + dist_loss_coeff * dist_loss
    else:
        return loss.mean()


def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "none",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def _final_aggregation(
        means_x: Tensor,
        means_y: Tensor,
        vars_x: Tensor,
        vars_y: Tensor,
        corrs_xy: Tensor,
        nbs: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Aggregate the statistics from multiple devices.

    Formula taken from here: `Aggregate the statistics from multiple devices`_
    """
    # assert len(means_x) > 1 and len(means_y) > 1 and len(vars_x) > 1 and len(vars_y) > 1 and len(corrs_xy) > 1
    mean_x, mean_y, var_x, var_y, corr_xy, nb = means_x[0], means_y[0], vars_x[0], vars_y[0], corrs_xy[0], nbs[0]
    for i in range(1, len(means_x)):
        mx2, my2, vx2, vy2, cxy2, n2 = means_x[i], means_y[i], vars_x[i], vars_y[i], corrs_xy[i], nbs[i]
        # vx2: batch_p_var, vy2: batch_l_var , cxy2: batch_bl_var
        delta_p_var = (vx2 + (mean_x - mx2) * (mean_x - mx2) * (
                nb * n2 / (nb + n2)))
        var_x += delta_p_var

        delta_l_var = (vy2 + (mean_y - my2) * (mean_y - my2) * (
                nb * n2 / (nb + n2)))
        var_y += delta_l_var

        delta_pl_var = (cxy2 + (mean_x - mx2) * (mean_y - my2) * (nb * n2) / (nb + n2))
        corr_xy += delta_pl_var

        nb += n2
        mean_x = (nb * mean_x + n2 * mx2) / nb
        mean_y = (nb * mean_y + n2 * my2) / nb

    return var_x, var_y, corr_xy, nb


def _corrcoeff_update(preds: Tensor,
                      target: Tensor,
                      mean_x: Tensor,
                      mean_y: Tensor,
                      var_x: Tensor,
                      var_y: Tensor,
                      corr_xy: Tensor,
                      n_prior: Tensor,
                      ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Updates and returns variables required to compute Pearson Correlation Coefficient. Checks for same shape of
    input tensors.

    Args:
        mean_x: current mean estimate of x tensor
        mean_y: current mean estimate of y tensor
        var_x: current variance estimate of x tensor
        var_y: current variance estimate of y tensor
        corr_xy: current covariance estimate between x and y tensor
        n_prior: current number of observed observations
    """
    # Data checking
    _check_same_shape(preds, target)

    preds = preds.squeeze()
    target = target.squeeze()
    if preds.ndim > 1 or target.ndim > 1:
        raise ValueError("Expected both predictions and target to be 1 dimensional tensors.")

    n_obs = preds.numel()
    batch_mean_preds = preds.mean()
    batch_mean_labels = target.mean()

    mx_new = (n_prior * mean_x + preds.mean() * n_obs) / (n_prior + n_obs)
    my_new = (n_prior * mean_y + target.mean() * n_obs) / (n_prior + n_obs)

    batch_p_var = ((preds - batch_mean_preds) * (preds - batch_mean_preds)).sum()
    delta_p_var = (batch_p_var + (mean_x - batch_mean_preds) * (mean_x - batch_mean_preds) * (
            n_prior * n_obs / (n_prior + n_obs)))
    var_x += delta_p_var

    batch_l_var = ((target - batch_mean_labels) * (target - batch_mean_labels)).sum()
    delta_l_var = (batch_l_var + (mean_y - batch_mean_labels) * (mean_y - batch_mean_labels) * (
            n_prior * n_obs / (n_prior + n_obs)))
    var_y += delta_l_var

    batch_pl_corr = ((preds - batch_mean_preds) * (target - batch_mean_labels)).sum()
    delta_pl_corr = (batch_pl_corr + (mean_x - batch_mean_preds) * (mean_y - batch_mean_labels) * (
            n_prior * n_obs / (n_prior + n_obs)))
    corr_xy += delta_pl_corr

    n_prior += n_obs

    mean_x = mx_new
    mean_y = my_new

    return mean_x, mean_y, var_x, var_y, corr_xy, n_prior


class ConCorrCoef(torchmetrics.Metric):
    """
    Based on: https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/regression/pearson.py
    """

    def __init__(self, num_classes=2, **kwargs):
        super(ConCorrCoef, self).__init__(**kwargs)
        self.add_state('mean_x', default=torch.tensor([0.] * num_classes), dist_reduce_fx=None)
        self.add_state('mean_y', default=torch.tensor([0.] * num_classes), dist_reduce_fx=None)
        self.add_state('var_x', default=torch.tensor([0.] * num_classes), dist_reduce_fx=None)
        self.add_state('var_y', default=torch.tensor([0.] * num_classes), dist_reduce_fx=None)
        self.add_state('corr_xy', default=torch.tensor([0.] * num_classes), dist_reduce_fx=None)
        self.add_state('n_total', default=torch.tensor([0.] * num_classes), dist_reduce_fx=None)

        self.num_classes = num_classes

    def update(self, yhat, y):
        preds = torch.reshape(yhat, (-1, self.num_classes))
        target = torch.reshape(y, (-1, self.num_classes))

        for idx in range(self.num_classes):
            self.mean_x[idx], self.mean_y[idx], self.var_x[idx], self.var_y[idx], self.corr_xy[idx], self.n_total[
                idx] = _corrcoeff_update(
                preds[:, idx], target[:, idx], self.mean_x[idx], self.mean_y[idx], self.var_x[idx], self.var_y[idx],
                self.corr_xy[idx], self.n_total[idx]
            )

    def compute(self):
        """Computes pearson correlation coefficient over state."""

        if self.mean_x[0].numel() > 1:  # multiple devices, need further reduction
            var_x = [0] * self.num_classes
            var_y = [0] * self.num_classes
            mean_x = [0] * self.num_classes
            mean_y = [0] * self.num_classes
            corr_xy = [0] * self.num_classes
            n_total = [0] * self.num_classes

            for idx in range(self.num_classes):
                var_x[idx], var_y[idx], corr_xy[idx], n_total[idx] = _final_aggregation(
                    self.mean_x[idx], self.mean_y[idx], self.var_x[idx], self.var_y[idx], self.corr_xy[idx],
                    self.n_total[idx]
                )
                mean_x[idx] = torch.mean(self.mean_x[idx])
                mean_y[idx] = torch.mean(self.mean_y[idx])
        else:
            var_x = self.var_x
            var_y = self.var_y
            mean_x = self.mean_x
            mean_y = self.mean_y
            corr_xy = self.corr_xy
            n_total = self.n_total

        ccc = [0] * self.num_classes
        for idx in range(self.num_classes):
            ccc[idx] = 2 * corr_xy[idx] / (
                        var_x[idx] + var_y[idx] + n_total[idx] * (mean_x[idx] - mean_y[idx]).square())

        return torch.mean(torch.stack(ccc))
