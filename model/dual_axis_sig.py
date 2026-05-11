# Output-level dual-axis fusion of two complete single-axis sub-models.
#
# Architecture:
#       y_temp = time_sig_fast.Model(configs)(x)        # full temporal-axis model
#       y_feat = feature_sig_fast.Model(configs)(x)     # full feature-axis model
#       alpha  = gate(pool(x_enc))                      # scalar in [0, 1]
#       y_out  = alpha * y_temp + (1 - alpha) * y_feat  # element-wise on [B, W, V]
#
# Endpoint equivalence (structural, not optimisation-dependent):
#       alpha == 1  =>  output is EXACTLY what time_sig_fast.Model would produce
#       alpha == 0  =>  output is EXACTLY what feature_sig_fast.Model would produce
#
# Cost: ~2x compute and ~2x parameters relative to either single-axis model.
#
# Two-stage training (handled by exp_long_term_forecasting.py via --two_stage_epochs):
#   stage 1: gate is free, both branches get gradient
#   stage 2: gate frozen at 0 or 1 (decided by stage-1 alpha mean), single branch
#
# Public surface for run.py:
#       --model dual_axis_sig --use_progressive_signature
#       --gate_init_alpha <float>      optional warm start
#       --dual_aux_lambda <float>      optional alpha=0/1 auxiliary loss
#       --two_stage_epochs <int>       optional two-stage commit

import os
os.environ['KERAS_BACKEND'] = 'torch'

import math
import torch
import torch.nn as nn

from model import time_sig_fast, feature_sig_fast


class AxisGate(nn.Module):
    """Per-sample scalar gate alpha in [0, 1] from pooled raw input.

    alpha weights the temporal branch; (1 - alpha) weights the feature branch.
    """

    def __init__(self, n_features, hidden=32, init_alpha=0.5):
        super().__init__()
        self.encode = nn.Linear(n_features, hidden)
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Initialise the final-linear bias so that sigmoid(bias) == init_alpha.
        init_alpha = float(min(max(init_alpha, 1e-4), 1 - 1e-4))
        bias_init = math.log(init_alpha / (1.0 - init_alpha))
        with torch.no_grad():
            self.head[-1].bias.fill_(bias_init)
            # Damp final-layer weights so the bias dominates early steps.
            self.head[-1].weight.mul_(0.1)

    def forward(self, x_enc):
        # x_enc: [B, L, V] -> pooled [B, V] -> alpha [B, 1, 1]
        pooled = x_enc.mean(dim=1)            # [B, V]
        h = self.encode(pooled)               # [B, hidden]
        a = torch.sigmoid(self.head(h))       # [B, 1]
        return a.unsqueeze(-1)                # [B, 1, 1] -- broadcasts over W and V


class Model(nn.Module):
    """Two complete single-axis sub-models combined at the output level."""

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.gate_init_alpha = getattr(configs, 'gate_init_alpha', 0.5)

        # Two complete sub-models running in parallel.
        # Each handles its own normalisation / embedding / sig / mamba / projection
        # and returns [B, pred_len, V] in the original (de-normalised) space.
        self.branch_temp = time_sig_fast.Model(configs)
        self.branch_feat = feature_sig_fast.Model(configs)

        # Gate operates on the raw V-dim input; portable across datasets because
        # we instantiate Linear(enc_in, hidden) per dataset.
        self.gate = AxisGate(configs.enc_in, init_alpha=self.gate_init_alpha)

        # Set during forward() for training-time monitoring.
        self.last_alpha = None

        if getattr(configs, 'use_progressive_signature', False):
            print(f"dual_axis_sig (output-level fusion) initialised: "
                  f"gate_init_alpha={self.gate_init_alpha}, "
                  f"branches=time_sig_fast + feature_sig_fast")

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,
                force_alpha=None):
        # Stage-2 / aux-loss fast path: when alpha is exactly pinned to an
        # endpoint, skip the unused branch entirely so compute matches a
        # single-axis model.
        if force_alpha is not None:
            fa = float(force_alpha)
            if fa >= 1.0 - 1e-9:
                return self.branch_temp(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if fa <= 1e-9:
                return self.branch_feat(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # Anything else still needs both branches for the convex combo.
            y_temp = self.branch_temp(x_enc, x_mark_enc, x_dec, x_mark_dec)
            y_feat = self.branch_feat(x_enc, x_mark_enc, x_dec, x_mark_dec)
            B = x_enc.shape[0]
            alpha = torch.full((B, 1, 1), fa,
                               device=y_temp.device, dtype=y_temp.dtype)
            return alpha * y_temp + (1.0 - alpha) * y_feat

        # Stage-1 / free-routing path: both branches run, gate decides.
        y_temp = self.branch_temp(x_enc, x_mark_enc, x_dec, x_mark_dec)
        y_feat = self.branch_feat(x_enc, x_mark_enc, x_dec, x_mark_dec)
        alpha = self.gate(x_enc).to(y_temp.dtype)
        self.last_alpha = alpha.detach().mean()
        return alpha * y_temp + (1.0 - alpha) * y_feat
