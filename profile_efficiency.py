"""Profile per-component time and GPU memory for time_sig_fast / feature_sig_fast.

Mirrors the args used by scripts/multivariate_forecasting/PEMS/siggg_03.sh
(PEMS03 96->12) and scripts/multivariate_forecasting/ETT/sigg_ettm2.sh
(ETTm2 96->96), then runs ~50 timed iterations, breaking the wall-clock down
into:
    transfer        : H2D copy of the batch
    embedding       : DataEmbedding_inverted
    sig_reduce      : value_head / pca_projection (channel reduction)
    sig_path        : time-augmentation + concat (path build)
    sig_call        : keras_sig.signature(...) (the kernel/algorithm)
    sig_project     : R^{sig_dim} -> R^channels linear + broadcast
    mamba_blocks    : forward through Mamba encoder layers
    other_forward   : projector + denorm + misc
    backward        : loss.backward()
    optimizer       : optim.step()

Outputs JSON to profile_results_<tag>.json.

Run:
    python profile_efficiency.py --config pems03
    python profile_efficiency.py --config ettm2
"""

import os
os.environ['KERAS_BACKEND'] = 'torch'

import argparse
import json
import time
import contextlib
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from data_provider.data_factory import data_provider
from model import time_sig_fast, feature_sig_fast, S_Mamba, iTransformer, Autoformer


# ---------- per-component CUDA timer ------------------------------------- #

class CudaTimer:
    """Accumulates time per labeled section using CUDA events."""

    def __init__(self):
        self.totals = {}
        self.counts = {}

    @contextlib.contextmanager
    def section(self, name):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end)
            self.totals[name] = self.totals.get(name, 0.0) + ms
            self.counts[name] = self.counts.get(name, 0) + 1

    def reset(self):
        self.totals.clear()
        self.counts.clear()

    def per_iter(self):
        return {k: self.totals[k] / max(1, self.counts[k]) for k in self.totals}


# ---------- monkey-patches that route through the timer ----------------- #

def patch_time_sig(module, timer):
    """Wrap TimeWindowProgressiveSignatureModule.forward with section timing."""
    from keras_sig import signature

    orig_pca = module.pca_projection
    orig_proj = module.projection

    def timed_forward(x):
        B, L, N = x.shape
        device = x.device

        with timer.section("sig_reduce"):
            x_red = orig_pca(x)

        with timer.section("sig_path"):
            t = torch.linspace(0, 1, L, device=device).view(1, L, 1).expand(B, -1, -1)
            full_path = torch.cat([t, x_red], dim=-1)

        with timer.section("sig_call"):
            all_sigs = signature(full_path, depth=module.signature_depth, stream=True)

        with timer.section("sig_project"):
            sig_len = all_sigs.shape[1]
            if sig_len == L - 1:
                pad = torch.zeros(B, 1, all_sigs.shape[-1], device=device,
                                  dtype=all_sigs.dtype)
                all_sigs = torch.cat([pad, all_sigs], dim=1)
            out = torch.zeros(B, L, module.n_features, device=device)
            for start in range(0, L, module.window_size):
                end = min(start + module.window_size, L)
                if end <= 2:
                    continue
                endpoint_sig = all_sigs[:, end - 1, :]
                proj = orig_proj(endpoint_sig)
                out[:, start:end, :] = proj.unsqueeze(1)
            if module.use_position_encoding and L <= module.pos_encoding.shape[1]:
                out = out + module.pos_encoding[:, :L, :]

        module.last_sig = out.detach()
        return out

    module.forward = timed_forward


def patch_feature_sig(module, timer):
    """Wrap ProgressiveSignatureModule.forward with section timing."""
    from keras_sig import signature

    orig_value_head = module.value_head
    orig_proj = module.projection

    def timed_forward(x):
        B, V_axis, C = x.shape
        device = x.device

        with timer.section("sig_reduce"):
            x_red = orig_value_head(x)

        with timer.section("sig_path"):
            if module.use_time_augmentation:
                t = torch.linspace(0, 1, V_axis, device=device).view(1, V_axis, 1).expand(B, -1, -1)
                full_path = torch.cat([t, x_red], dim=-1)
            else:
                full_path = x_red

        with timer.section("sig_call"):
            all_sigs = signature(full_path, depth=module.signature_depth, stream=True)

        with timer.section("sig_project"):
            sig_len = all_sigs.shape[1]
            if sig_len == V_axis - 1:
                pad = torch.zeros(B, 1, all_sigs.shape[-1], device=device,
                                  dtype=all_sigs.dtype)
                all_sigs = torch.cat([pad, all_sigs], dim=1)
            out = torch.zeros(B, V_axis, C, device=device)
            for start in range(0, V_axis, module.window_size):
                end = min(start + module.window_size, V_axis)
                if end <= 2:
                    continue
                endpoint_sig = all_sigs[:, end - 1, :]
                proj = orig_proj(endpoint_sig)
                out[:, start:end, :] = proj.unsqueeze(1)
            if module.use_position_encoding and V_axis <= module.pos_encoding.shape[1]:
                out = out + module.pos_encoding[:, :V_axis, :]

        return out

    module.forward = timed_forward


# ---------- argument presets that mirror the .sh scripts ----------------- #

def make_args(config):
    base = dict(
        is_training=1, model_id="prof", des="Exp", features="M",
        embed="timeF", freq="h", checkpoints="./checkpoints/",
        seq_len=96, label_len=48, target="OT", inverse=False,
        moving_avg=25, factor=1, dropout=0.1, distil=True,
        n_heads=8, d_layers=1, num_workers=4, itr=1, train_epochs=1,
        patience=3, learning_rate=1e-4, lradj="type1", loss="MSE",
        use_amp=False, output_attention=False, do_predict=False,
        use_signature=False, use_progressive_signature=True,
        signature_depth=3, signature_fusion="concat",
        signature_window_size=8, signature_r2=3, signature_pca_dim=3,
        sigkernel_signature_depth=3, signature_cache_size=1000,
        history_buffer_size=50, sig_window_size=10,
        sig_temperature=1.0, use_sigkernel_loss=False,
        use_gpu=True, gpu=0, use_multi_gpu=False, devices="0",
        exp_name="MTSF", channel_independence=False,
        class_strategy="projection",
        target_root_path="./data/electricity/", target_data_path="electricity.csv",
        efficient_training=False, use_norm=True, partial_start_index=0,
        use_sig_loss=False, lambda_sig=1e-4, sig_loss_depth=2,
        sig_loss_leadlag=False, gate_init_alpha=0.5, dual_aux_lambda=0.3,
        two_stage_epochs=0, activation="gelu", batch_size=16,
        signature_time_aug=True,
    )

    # config keys: pems03 / pems03_smamba / pems03_itrans / ettm2 / ettm2_smamba / ettm2_itrans
    if config.startswith("pems03"):
        base.update(dict(
            model="time_sig_fast",
            data="PEMS",
            root_path="./dataset/PEMS/",
            data_path="PEMS03.npz",
            pred_len=12, e_layers=4,
            enc_in=358, dec_in=358, c_out=358,
            d_model=512, d_ff=512, d_state=32,
            signature_pca_dim=3,
        ))
        if config == "pems03_smamba":
            base.update(dict(model="S_Mamba", use_progressive_signature=False))
        elif config == "pems03_itrans":
            base.update(dict(model="iTransformer", use_progressive_signature=False))
        elif config == "pems03_auto":
            base.update(dict(model="Autoformer", use_progressive_signature=False))
    elif config.startswith("ettm2"):
        base.update(dict(
            model="feature_sig_fast",
            data="ETTm2",
            root_path="./dataset/ETT-small/",
            data_path="ETTm2.csv",
            pred_len=96, e_layers=4,
            enc_in=7, dec_in=7, c_out=7,
            d_model=256, d_ff=256, d_state=2,
            signature_pca_dim=5,
        ))
        if config == "ettm2_smamba":
            base.update(dict(model="S_Mamba", use_progressive_signature=False))
        elif config == "ettm2_itrans":
            base.update(dict(model="iTransformer", use_progressive_signature=False))
        elif config == "ettm2_auto":
            base.update(dict(model="Autoformer", use_progressive_signature=False))
    else:
        raise ValueError(config)

    return SimpleNamespace(**base)


# ---------- main profiling routine --------------------------------------- #

def profile(config, n_warmup=10, n_iters=50):
    args = make_args(config)
    args.use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0")

    # data
    train_data, train_loader = data_provider(args, flag="train")

    # model
    if args.model == "time_sig_fast":
        model = time_sig_fast.Model(args).float().to(device)
    elif args.model == "feature_sig_fast":
        model = feature_sig_fast.Model(args).float().to(device)
    elif args.model == "S_Mamba":
        model = S_Mamba.Model(args).float().to(device)
    elif args.model == "iTransformer":
        model = iTransformer.Model(args).float().to(device)
    elif args.model == "Autoformer":
        model = Autoformer.Model(args).float().to(device)
    else:
        raise ValueError(args.model)

    timer = CudaTimer()
    if args.model == "time_sig_fast":
        patch_time_sig(model.time_signature_module, timer)
    elif args.model == "feature_sig_fast":
        first_layer = model.encoder.attn_layers[0]
        psm = first_layer.attention.progressive_signature_module
        patch_feature_sig(psm, timer)

    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    crit = nn.MSELoss()

    # also time mamba blocks via forward hooks on Mamba modules
    from mamba_ssm import Mamba

    mamba_handles = []

    def add_mamba_hooks():
        def pre(_m, _inp):
            torch.cuda.synchronize()
            _m._t0 = time.perf_counter()

        def post(_m, _inp, _out):
            torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - _m._t0) * 1000.0
            timer.totals["mamba_blocks"] = timer.totals.get("mamba_blocks", 0.0) + dt_ms
            timer.counts["mamba_blocks"] = timer.counts.get("mamba_blocks", 0) + 1

        for mod in model.modules():
            if isinstance(mod, Mamba):
                mamba_handles.append(mod.register_forward_pre_hook(pre))
                mamba_handles.append(mod.register_forward_hook(post))

    add_mamba_hooks()

    # iterator that reproduces exp_long_term_forecasting.train inner loop
    def step(batch):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch

        with timer.section("transfer"):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

        with timer.section("forward_total"):
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == "MS" else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y2 = batch_y[:, -args.pred_len:, f_dim:]
            loss = crit(outputs, batch_y2)

        with timer.section("backward"):
            optim.zero_grad()
            loss.backward()

        with timer.section("optimizer"):
            optim.step()

        return loss.item()

    # warmup
    print(f"[{config}] warming up ({n_warmup} iters)...")
    it = iter(train_loader)
    for _ in range(n_warmup):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        step(batch)

    # measured
    timer.reset()
    torch.cuda.reset_peak_memory_stats(device)
    print(f"[{config}] measuring ({n_iters} iters)...")
    losses = []
    iter_starts = []
    iter_ends = []

    t0 = time.perf_counter()
    for i in range(n_iters):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        torch.cuda.synchronize()
        iter_starts.append(time.perf_counter())
        l = step(batch)
        torch.cuda.synchronize()
        iter_ends.append(time.perf_counter())
        losses.append(l)
    total_wall = time.perf_counter() - t0

    peak_mem_bytes = torch.cuda.max_memory_allocated(device)
    peak_reserved_bytes = torch.cuda.max_memory_reserved(device)

    per_iter = timer.per_iter()
    iter_durations_ms = [(e - s) * 1000.0 for s, e in zip(iter_starts, iter_ends)]
    measured_iter_ms = float(np.mean(iter_durations_ms))

    # derived: "other_forward" = forward_total - (transfer is outside) - sum(sig_*) - mamba
    sig_sum = sum(per_iter.get(k, 0.0) for k in ("sig_reduce", "sig_path", "sig_call", "sig_project"))
    other_fwd = max(0.0, per_iter.get("forward_total", 0.0) - sig_sum - per_iter.get("mamba_blocks", 0.0))

    result = {
        "config": config,
        "n_iters": n_iters,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "enc_in": args.enc_in,
        "d_model": args.d_model,
        "e_layers": args.e_layers,
        "signature_depth": args.signature_depth,
        "signature_window_size": args.signature_window_size,
        "loss_mean": float(np.mean(losses)),
        "iter_ms_mean": measured_iter_ms,
        "iter_ms_std": float(np.std(iter_durations_ms)),
        "total_wall_s": total_wall,
        "peak_gpu_mem_gb": peak_mem_bytes / (1024 ** 3),
        "peak_gpu_reserved_gb": peak_reserved_bytes / (1024 ** 3),
        "per_component_ms": {
            **per_iter,
            "other_forward": other_fwd,
        },
    }

    out_path = f"profile_results_{config}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"[{config}] wrote {out_path}")

    for h in mamba_handles:
        h.remove()
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",
                   choices=["pems03", "ettm2", "both",
                            "pems03_smamba", "pems03_itrans", "pems03_auto",
                            "ettm2_smamba", "ettm2_itrans", "ettm2_auto", "all"],
                   default="both")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    cli = p.parse_args()

    if cli.config == "both":
        configs = ["pems03", "ettm2"]
    elif cli.config == "all":
        configs = ["pems03", "pems03_smamba", "pems03_itrans",
                   "ettm2", "ettm2_smamba", "ettm2_itrans"]
    else:
        configs = [cli.config]
    for c in configs:
        profile(c, n_warmup=cli.warmup, n_iters=cli.iters)
        torch.cuda.empty_cache()
