import os
os.environ['KERAS_BACKEND'] = 'torch'

import torch
import torch.nn as nn
from layers.Mamba_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding_inverted
from mamba_ssm import Mamba


class ProgressiveDeepSetsModule(nn.Module):
    def __init__(self, channels, window_size=8, hidden=16, r2=3,
                 use_position_encoding=True):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.r2 = r2

        # W_{r2} : R^D -> R^{r2}, the learnable channel-reduction in the paper
        self.value_head = nn.Linear(channels, r2, bias=True)

        self.phi = nn.Sequential(
            nn.Linear(r2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.rho = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

        self.use_position_encoding = use_position_encoding
        if self.use_position_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, 1000, channels) * 0.1)

        print(f"ProgressiveDeepSetsModule (paper-aligned): "
              f"r2={r2}, hidden={hidden}, channels={channels}")

    def forward(self, x):
        # x: [B, V, D]
        B, V_axis, C = x.shape
        device = x.device

        # ---- Step 1: learnable channel reduction ----
        x_red = self.value_head(x)                                  # [B, V, r2]

        out = torch.zeros(B, V_axis, C, device=device)
        for start in range(0, V_axis, self.window_size):
            end = min(start + self.window_size, V_axis)
            if end <= 2:
                continue
            slice_ = x_red[:, :end, :]                              # [B, end, r2]
            phi_out = self.phi(slice_)                              # [B, end, hidden]
            pooled = phi_out.sum(dim=1)                             # [B, hidden] (PI in V)
            feat = self.rho(pooled)                                 # [B, C]
            out[:, start:end, :] = feat.unsqueeze(1)

        if self.use_position_encoding and V_axis <= self.pos_encoding.shape[1]:
            out = out + self.pos_encoding[:, :V_axis, :]
        return out


class ProgressiveDeepSetsEnhancedMamba(nn.Module):
    def __init__(self, d_model, d_state, window_size, fusion_method,
                 hidden=16, r2=3):
        super().__init__()
        self.fusion_method = fusion_method
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=2, expand=1)
        self.module = ProgressiveDeepSetsModule(
            channels=d_model, window_size=window_size, hidden=hidden, r2=r2,
        )
        if fusion_method == 'concat':
            self.fusion = nn.Linear(d_model * 2, d_model)
        elif fusion_method == 'gated':
            self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        elif fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
            self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        m = self.mamba(x)
        f = self.module(x)
        if self.fusion_method == 'concat':
            return self.fusion(torch.cat([m, f], dim=-1))
        elif self.fusion_method == 'gated':
            g = self.gate(torch.cat([m, f], dim=-1))
            return g * m + (1 - g) * f
        elif self.fusion_method == 'attention':
            a, _ = self.attention(m, f, f)
            return self.norm(m + a)
        return m + f


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.use_progressive_signature = getattr(configs, 'use_progressive_signature', False)
        self.signature_window_size = getattr(configs, 'signature_window_size', 8)
        self.signature_fusion = getattr(configs, 'signature_fusion', 'concat')
        self.signature_r2 = getattr(configs, 'signature_r2', 3)
        self.class_strategy = configs.class_strategy

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        encoder_layers = []
        for l in range(configs.e_layers):
            if self.use_progressive_signature and l == 0:
                first_attn = ProgressiveDeepSetsEnhancedMamba(
                    d_model=configs.d_model, d_state=configs.d_state,
                    window_size=self.signature_window_size,
                    fusion_method=self.signature_fusion,
                    r2=self.signature_r2,
                )
            else:
                first_attn = Mamba(d_model=configs.d_model, d_state=configs.d_state, d_conv=2, expand=1)
            second_attn = Mamba(d_model=configs.d_model, d_state=configs.d_state, d_conv=2, expand=1)
            encoder_layers.append(EncoderLayer(
                first_attn, second_attn, configs.d_model, configs.d_ff,
                dropout=configs.dropout, activation=configs.activation,
            ))

        self.encoder = Encoder(encoder_layers, norm_layer=torch.nn.LayerNorm(configs.d_model))
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.use_progressive_signature:
            print(f"feature_deepsets_fast (paper-aligned) initialised: "
                  f"r2={self.signature_r2}, window={self.signature_window_size}")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]
        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]
