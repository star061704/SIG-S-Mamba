# Truly fast temporal-axis signature, drop-in replacement for time_sig.py.
#
# Mirrors the strategy in feature_sig_fast.py but along the time (L) axis on
# raw input [B, L, N]:
#   - keras_sig.signature(stream=True) ONCE for all prefixes of the full
#     time-augmented path -> [B, L, sig_dim]
#   - gather window endpoints and broadcast within each window block
#   - fallback: per-window batched calls (no per-sample Python loop)
#
# Public surface (constructor signature, parameter names, output shape) is
# identical to time_sig.TimeWindowProgressiveSignatureModule, so a checkpoint
# trained with the original module can still be loaded here.

import os
os.environ['KERAS_BACKEND'] = 'torch'

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Mamba_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding_inverted
from mamba_ssm import Mamba

from keras_sig import signature


class TimeWindowProgressiveSignatureModule(nn.Module):
    """Progressive prefix signatures along the time (L) axis -- fast variant.

    Input  x : [B, L, N]   raw multivariate series
    Output o : [B, L, N]   per-step signature features, broadcast in blocks
    """

    def __init__(self, n_features, signature_depth=3, window_size=8, pca_dim=3):
        super().__init__()
        self.n_features = n_features
        self.signature_depth = signature_depth
        self.window_size = window_size
        self.pca_dim = pca_dim

        # Learnable channel reduction; legacy name kept for state-dict compat.
        self.pca_projection = nn.Linear(n_features, pca_dim)

        path_dim = pca_dim + 1  # +1 for time
        self.signature_dim = sum(path_dim ** i
                                 for i in range(1, signature_depth + 1))

        self.projection = nn.Linear(self.signature_dim, n_features)

        self.use_position_encoding = True
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, n_features) * 0.1)

        # Decided lazily on first forward, based on what keras_sig returns.
        self._stream_supported = None  # None=untried, True=stream, False=fallback

        print(f"TimeWindowProgressiveSignatureModule (fast, stream=True): "
              f"n_features={n_features}, pca_dim={pca_dim}, "
              f"window_size={window_size}, depth={signature_depth}, "
              f"sig_dim={self.signature_dim}")

    def forward(self, x):
        # x: [B, L, N]
        B, L, N = x.shape
        device = x.device

        # 1) batched channel reduction (no per-sample Python loop)
        x_red = self.pca_projection(x)                              # [B, L, pca_dim]

        # 2) build the full time-augmented path once
        t = torch.linspace(0, 1, L, device=device).view(1, L, 1).expand(B, -1, -1)
        full_path = torch.cat([t, x_red], dim=-1)                   # [B, L, pca_dim+1]

        # 3) ONE signature call (stream=True) for all prefixes
        if self._stream_supported is not False:
            try:
                all_sigs = signature(full_path, depth=self.signature_depth, stream=True)
                if all_sigs.dim() != 3 or all_sigs.shape[0] != B \
                        or all_sigs.shape[-1] != self.signature_dim:
                    raise RuntimeError(
                        f"unexpected stream output shape {tuple(all_sigs.shape)}; "
                        f"expected (B={B}, L<={L}, sig_dim={self.signature_dim})"
                    )
                self._stream_supported = True
                out = self._gather_blocks(all_sigs, B, L, device)
                self.last_sig = out.detach()
                return out
            except (TypeError, RuntimeError, Exception) as e:
                if self._stream_supported is None:
                    print(f"[stream signature unavailable: {e}] -> using window fallback")
                self._stream_supported = False

        # 4) Fallback: per-window batched calls (still no per-sample loop)
        out = self._forward_window_loop(x_red, B, L, device)
        self.last_sig = out.detach()
        return out

    def _gather_blocks(self, all_sigs, B, L, device):
        """Index window endpoints from a streamed signature tensor and broadcast."""
        sig_len = all_sigs.shape[1]
        if sig_len == L - 1:
            pad = torch.zeros(B, 1, all_sigs.shape[-1], device=device,
                              dtype=all_sigs.dtype)
            all_sigs_padded = torch.cat([pad, all_sigs], dim=1)
        elif sig_len == L:
            all_sigs_padded = all_sigs
        else:
            raise RuntimeError(
                f"unexpected stream signature length {sig_len} for L={L}"
            )

        out = torch.zeros(B, L, self.n_features, device=device)
        for start in range(0, L, self.window_size):
            end = min(start + self.window_size, L)
            if end <= 2:
                continue
            endpoint_sig = all_sigs_padded[:, end - 1, :]           # [B, sig_dim]
            proj = self.projection(endpoint_sig)                    # [B, n_features]
            out[:, start:end, :] = proj.unsqueeze(1)

        if self.use_position_encoding and L <= self.pos_encoding.shape[1]:
            out = out + self.pos_encoding[:, :L, :]
        return out

    def _forward_window_loop(self, x_red, B, L, device):
        """Fallback: per-window batched calls; no per-sample Python loop."""
        out = torch.zeros(B, L, self.n_features, device=device)
        for start in range(0, L, self.window_size):
            end = min(start + self.window_size, L)
            if end <= 2:
                continue
            prefix = x_red[:, :end, :]
            t = torch.linspace(0, 1, end, device=device).view(1, end, 1).expand(B, -1, -1)
            path = torch.cat([t, prefix], dim=-1)
            try:
                sig = signature(path, depth=self.signature_depth)   # [B, sig_dim]
                proj = self.projection(sig)                          # [B, n_features]
                out[:, start:end, :] = proj.unsqueeze(1)
            except Exception as e:
                print(f"signature failed at window [{start}:{end}]: {e}")
        if self.use_position_encoding and L <= self.pos_encoding.shape[1]:
            out = out + self.pos_encoding[:, :L, :]
        return out


class ProgressiveSignatureEnhancedMamba(nn.Module):
    """Mamba block fused with externally-computed signature features."""
    def __init__(self, d_model, d_state=32, signature_depth=3, window_size=8,
                 fusion_method='concat'):
        super().__init__()
        self.fusion_method = fusion_method
        self.d_model = d_model
        self.window_size = window_size

        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=2, expand=1)

        if fusion_method == 'concat':
            self.fusion = nn.Linear(d_model * 2, d_model)
        elif fusion_method == 'gated':
            self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        elif fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
        elif fusion_method == 'add':
            pass

        print(f"ProgressiveSignatureEnhancedMamba (time_sig_fast): "
              f"d_model={d_model}, fusion={fusion_method}")

    def forward(self, x, sig_features):
        mamba_out = self.mamba(x)

        if self.fusion_method == 'concat':
            return self.fusion(torch.cat([mamba_out, sig_features], dim=-1))
        elif self.fusion_method == 'gated':
            g = self.gate(torch.cat([mamba_out, sig_features], dim=-1))
            return g * mamba_out + (1 - g) * sig_features
        elif self.fusion_method == 'attention':
            attended, _ = self.attention(mamba_out, sig_features, sig_features)
            return self.norm(mamba_out + attended)
        return mamba_out + sig_features


class SignatureEnhancedEncoderLayer(nn.Module):
    """Encoder layer that routes signature features into the first attention."""
    def __init__(self, attention, attention_r, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention      # ProgressiveSignatureEnhancedMamba
        self.attention_r = attention_r  # plain Mamba
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, sig_features=None, attn_mask=None, tau=None, delta=None):
        if sig_features is not None and isinstance(self.attention, ProgressiveSignatureEnhancedMamba):
            new_x1 = self.attention(x, sig_features)
        else:
            new_x1 = self.attention(x)
        new_x2 = self.attention_r(x.flip(dims=[1])).flip(dims=[1])

        new_x = new_x1 + new_x2
        attn = 1

        x = x + new_x
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Model(nn.Module):
    """Time-axis progressive signature + S-Mamba (fast variant)."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        self.use_progressive_signature = getattr(configs, 'use_progressive_signature', False)
        self.signature_depth = getattr(configs, 'signature_depth', 3)
        self.signature_window_size = getattr(configs, 'signature_window_size', 8)
        self.signature_fusion = getattr(configs, 'signature_fusion', 'concat')
        self.signature_pca_dim = getattr(configs, 'signature_pca_dim', 3)
        self.class_strategy = configs.class_strategy

        if self.use_progressive_signature:
            self.time_signature_module = TimeWindowProgressiveSignatureModule(
                n_features=configs.enc_in,
                signature_depth=self.signature_depth,
                window_size=self.signature_window_size,
                pca_dim=self.signature_pca_dim,
            )

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed,
            configs.freq, configs.dropout,
        )

        if self.use_progressive_signature:
            self.sig_embedding = DataEmbedding_inverted(
                configs.seq_len, configs.d_model, configs.embed,
                configs.freq, configs.dropout,
            )

        encoder_layers = []
        for l in range(configs.e_layers):
            if self.use_progressive_signature and l == 0:
                first_attn = ProgressiveSignatureEnhancedMamba(
                    d_model=configs.d_model,
                    d_state=configs.d_state,
                    signature_depth=self.signature_depth,
                    window_size=self.signature_window_size,
                    fusion_method=self.signature_fusion,
                )
            else:
                first_attn = Mamba(
                    d_model=configs.d_model, d_state=configs.d_state, d_conv=2, expand=1,
                )
            second_attn = Mamba(
                d_model=configs.d_model, d_state=configs.d_state, d_conv=2, expand=1,
            )
            if self.use_progressive_signature and l == 0:
                encoder_layers.append(
                    SignatureEnhancedEncoderLayer(
                        first_attn, second_attn, configs.d_model, configs.d_ff,
                        dropout=configs.dropout, activation=configs.activation,
                    )
                )
            else:
                encoder_layers.append(
                    EncoderLayer(
                        first_attn, second_attn, configs.d_model, configs.d_ff,
                        dropout=configs.dropout, activation=configs.activation,
                    )
                )

        self.encoder = Encoder(encoder_layers, norm_layer=torch.nn.LayerNorm(configs.d_model))
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        if self.use_progressive_signature:
            print(f"time_sig_fast initialised: depth={self.signature_depth}, "
                  f"window={self.signature_window_size}, pca_dim={self.signature_pca_dim}, "
                  f"fusion={self.signature_fusion}")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        sig_features = None
        if self.use_progressive_signature:
            sig_features = self.time_signature_module(x_enc)        # [B, L, N]

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            if sig_features is not None:
                sig_features = (sig_features - means) / stdev

        _, _, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, x_mark_enc)             # [B, N, d_model]
        sig_emb = None
        if sig_features is not None:
            sig_emb = self.sig_embedding(sig_features, x_mark_enc)  # [B, N, d_model]

        if sig_features is not None:
            enc_out, _ = self._encoder_with_signature(enc_out, sig_emb, attn_mask=None)
        else:
            enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def _encoder_with_signature(self, x, sig_features, attn_mask=None):
        attns = []
        if len(self.encoder.attn_layers) > 0:
            first_layer = self.encoder.attn_layers[0]
            if isinstance(first_layer, SignatureEnhancedEncoderLayer):
                x, attn = first_layer(x, sig_features, attn_mask=attn_mask)
            else:
                x, attn = first_layer(x, attn_mask=attn_mask)
            attns.append(attn)
            for attn_layer in self.encoder.attn_layers[1:]:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        if self.encoder.norm is not None:
            x = self.encoder.norm(x)
        return x, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]
