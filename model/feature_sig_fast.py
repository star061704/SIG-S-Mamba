# Truly fast feature-axis signature, paper-aligned.
#
# Strategy: keras_sig.signature with stream=True returns the prefix signature
# at every step in a single call -- this is exactly the Chen-identity-based
# incremental update the paper invokes. We compute signatures for the FULL
# V-axis path once, then index out the endpoints of each window. The previous
# fast version called signature() K times (once per window), each time
# recomputing prefixes from scratch; this version calls it once, total.
#
# Paper alignment:
#   - learnable W_{r2} : R^D -> R^{r2}  (replacing the fixed mean)
#   - time-augmented path of dimension r2+1
#   - signature_dim = sum_{i=1..M} (r2+1)^i, matching d_sig(r2+1, M)
#   - block-wise broadcast within C_k, matching S^{feat} = sum 1{v in C_k} S~_k

import os
os.environ['KERAS_BACKEND'] = 'torch'

import torch
import torch.nn as nn
from layers.Mamba_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding_inverted
from mamba_ssm import Mamba

from keras_sig import signature


class ProgressiveSignatureModule(nn.Module):
    """
    Progressive prefix signatures along the feature (V) axis.

    Input  x : [B, V, D]     -- after DataEmbedding_inverted, second axis is V
    Output o : [B, V, D]     -- per-position signature features, broadcast in blocks
    """

    def __init__(self, channels, signature_depth=3, window_size=8, r2=3,
                 use_time_augmentation=True, use_position_encoding=True):
        super().__init__()
        self.channels = channels
        self.signature_depth = signature_depth
        self.window_size = window_size
        self.r2 = r2
        self.use_time_augmentation = use_time_augmentation

        # W_{r2} : R^D -> R^{r2}, the learnable channel-reduction in the paper
        self.value_head = nn.Linear(channels, r2, bias=True)

        self.path_dim = r2 + 1 if use_time_augmentation else r2
        self.signature_dim = sum(self.path_dim ** i
                                 for i in range(1, signature_depth + 1))

        # P_B : R^{sig_dim} -> R^D
        self.projection = nn.Linear(self.signature_dim, channels)

        self.use_position_encoding = use_position_encoding
        if self.use_position_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, 1000, channels) * 0.1)

        # Decided lazily on first forward, based on what keras_sig actually returns.
        self._stream_supported = None  # None=untried, True=use stream, False=fallback

        print(f"ProgressiveSignatureModule (truly fast, stream=True): "
              f"depth={signature_depth}, window_size={window_size}, "
              f"r2={r2}, path_dim={self.path_dim}, "
              f"sig_dim={self.signature_dim}, channels={channels}")

    def forward(self, x):
        # x: [B, V, D]
        B, V_axis, C = x.shape
        device = x.device

        # ---- Step 1: learnable channel reduction (paper's W_{r2}) ----
        x_red = self.value_head(x)                                  # [B, V, r2]

        # ---- Step 2: build the full time-augmented path once ----
        if self.use_time_augmentation:
            t = torch.linspace(0, 1, V_axis, device=device)         # [V]
            t = t.view(1, V_axis, 1).expand(B, -1, -1)              # [B, V, 1]
            full_path = torch.cat([t, x_red], dim=-1)               # [B, V, r2+1]
        else:
            full_path = x_red                                        # [B, V, r2]

        # ---- Step 3: ONE signature call (stream=True) for all prefixes ----
        if self._stream_supported is not False:
            try:
                all_sigs = signature(full_path, depth=self.signature_depth, stream=True)
                # Validate shape
                if all_sigs.dim() != 3 or all_sigs.shape[0] != B \
                        or all_sigs.shape[-1] != self.signature_dim:
                    raise RuntimeError(
                        f"unexpected stream output shape {tuple(all_sigs.shape)}; "
                        f"expected (B={B}, V<={V_axis}, sig_dim={self.signature_dim})"
                    )
                self._stream_supported = True
                return self._gather_blocks(all_sigs, B, V_axis, C, device)
            except (TypeError, RuntimeError, Exception) as e:
                if self._stream_supported is None:
                    print(f"[stream signature unavailable: {e}] -> using window fallback")
                self._stream_supported = False

        # ---- Fallback: per-window batched calls ----
        return self._forward_window_loop(x_red, B, V_axis, C, device)

    def _gather_blocks(self, all_sigs, B, V_axis, C, device):
        """Index window endpoints from a streamed signature tensor and broadcast."""
        # all_sigs: [B, sig_len, sig_dim]; sig_len is V or V-1 depending on convention
        sig_len = all_sigs.shape[1]
        if sig_len == V_axis - 1:
            # Pad-front so index k-1 corresponds to signature of prefix [0:k]
            pad = torch.zeros(B, 1, all_sigs.shape[-1], device=device,
                              dtype=all_sigs.dtype)
            all_sigs_padded = torch.cat([pad, all_sigs], dim=1)
        elif sig_len == V_axis:
            all_sigs_padded = all_sigs
        else:
            raise RuntimeError(
                f"Unexpected stream signature length {sig_len} for V={V_axis}"
            )

        out = torch.zeros(B, V_axis, C, device=device)
        for start in range(0, V_axis, self.window_size):
            end = min(start + self.window_size, V_axis)
            if end <= 2:
                continue
            endpoint_sig = all_sigs_padded[:, end - 1, :]           # [B, sig_dim]
            proj = self.projection(endpoint_sig)                    # [B, C]
            out[:, start:end, :] = proj.unsqueeze(1)

        if self.use_position_encoding and V_axis <= self.pos_encoding.shape[1]:
            out = out + self.pos_encoding[:, :V_axis, :]
        return out

    def _forward_window_loop(self, x_red, B, V_axis, C, device):
        """Fallback: per-window batched calls (the previous fast variant)."""
        out = torch.zeros(B, V_axis, C, device=device)
        for start in range(0, V_axis, self.window_size):
            end = min(start + self.window_size, V_axis)
            if end <= 2:
                continue
            prefix = x_red[:, :end, :]
            if self.use_time_augmentation:
                t = torch.linspace(0, 1, end, device=device).view(1, end, 1).expand(B, -1, -1)
                path = torch.cat([t, prefix], dim=-1)
            else:
                path = prefix
            try:
                sig = signature(path, depth=self.signature_depth)
                proj = self.projection(sig)
                out[:, start:end, :] = proj.unsqueeze(1)
            except Exception as e:
                print(f"Signature failed at window [{start}:{end}]: {e}")
        if self.use_position_encoding and V_axis <= self.pos_encoding.shape[1]:
            out = out + self.pos_encoding[:, :V_axis, :]
        return out


class ProgressiveSignatureEnhancedMamba(nn.Module):
    def __init__(self, d_model, d_state=32, signature_depth=3, window_size=8,
                 fusion_method='concat', r2=3, use_time_augmentation=True):
        super().__init__()
        self.fusion_method = fusion_method
        self.d_model = d_model
        self.window_size = window_size

        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=2, expand=1)
        self.progressive_signature_module = ProgressiveSignatureModule(
            channels=d_model,
            signature_depth=signature_depth,
            window_size=window_size,
            r2=r2,
            use_time_augmentation=use_time_augmentation,
        )

        if fusion_method == 'concat':
            self.fusion = nn.Linear(d_model * 2, d_model)
        elif fusion_method == 'gated':
            self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        elif fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
        elif fusion_method == 'add':
            pass

        print(f"ProgressiveSignatureEnhancedMamba (truly fast): "
              f"d_model={d_model}, window_size={window_size}, fusion={fusion_method}")

    def forward(self, x):
        m = self.mamba(x)
        s = self.progressive_signature_module(x)
        self.last_sig = s.detach()
        self.last_mamba = m.detach()

        if self.fusion_method == 'concat':
            return self.fusion(torch.cat([m, s], dim=-1))
        elif self.fusion_method == 'gated':
            g = self.gate(torch.cat([m, s], dim=-1))
            return g * m + (1 - g) * s
        elif self.fusion_method == 'attention':
            a, _ = self.attention(m, s, s)
            return self.norm(m + a)
        return m + s


class Model(nn.Module):
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
        self.signature_r2 = getattr(configs, 'signature_r2',3)
        self.signature_time_aug = getattr(configs, 'signature_time_aug', True)
        self.class_strategy = configs.class_strategy

        self.enc_embedding = DataEmbedding_inverted(
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
                    r2=self.signature_r2,
                    use_time_augmentation=self.signature_time_aug,
                )
            else:
                first_attn = Mamba(
                    d_model=configs.d_model, d_state=configs.d_state, d_conv=2, expand=1,
                )
            second_attn = Mamba(
                d_model=configs.d_model, d_state=configs.d_state, d_conv=2, expand=1,
            )
            encoder_layers.append(
                EncoderLayer(
                    first_attn, second_attn, configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                )
            )

        self.encoder = Encoder(encoder_layers, norm_layer=torch.nn.LayerNorm(configs.d_model))
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        if self.use_progressive_signature:
            print(f"feature_sig_fast (truly fast, paper-aligned) initialised: "
                  f"r2={self.signature_r2}, time_aug={self.signature_time_aug}, "
                  f"depth={self.signature_depth}, window={self.signature_window_size}")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, x_mark_enc)            # [B, V, D]
        enc_out, _ = self.encoder(enc_out, attn_mask=None)         # [B, V, D]
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]