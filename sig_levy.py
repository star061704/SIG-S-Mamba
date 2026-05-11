"""Per-window level-1 and Levy-area distributions.

Tests the prediction:
  * Cycle-dominated windows (ETT) -> level-1 ||S^(1)|| concentrates near 0
    (path returns near origin), and the antisymmetric part of level-2
    (= 2 x Levy area) has its variance across windows averaged out over
    many repeating cycles.
  * Trend/single-cycle-dominated windows (PEMS at L < ~3P) -> ||S^(1)||
    spreads out (each window samples a different trend slope) and Levy
    area varies sharply between windows (each catches a different phase
    of the long cycle).

For each dataset we use the same pipeline as model/time_sig_fast.py:
  PCA-3 channel reduction -> time-augmentation -> [W, L, 4] path bag.
We compute per window:
  S1[i]   = X[L-1, i] - X[0, i]                     (4-vector, level 1)
  S2[i,j] = level-2 signature                       (4x4)
  sym  = (S2 + S2.T) / 2     (= 1/2 * outer(S1, S1), endpoint-determined)
  anti = (S2 - S2.T) / 2     (= Levy area in plane (i,j))
Then aggregate ||S1||_2, ||sym||_F, ||anti||_F across windows and dataset.

Outputs:
  sig_levy_results.csv
  figs/sig_levy/hist_S1.png        overlaid ||S1|| histograms
  figs/sig_levy/hist_anti.png      overlaid ||S2_anti|| histograms
  figs/sig_levy/scatter_sym_anti.png   sym vs anti per window per dataset
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import iisignature


CASES = [
    # name, path, kind, L, P    -> cycles_per_window = L/P
    ("ETTh1",  "./dataset/ETT-small/ETTh1.csv", "ett_hour",   96, 24),    # 4.00
    ("ETTh2",  "./dataset/ETT-small/ETTh2.csv", "ett_hour",   96, 24),    # 4.00
    ("ETTm1",  "./dataset/ETT-small/ETTm1.csv", "ett_minute", 96, 96),    # 1.00
    ("ETTm2",  "./dataset/ETT-small/ETTm2.csv", "ett_minute", 96, 96),    # 1.00
    ("PEMS03", "./dataset/PEMS/PEMS03.npz",     "pems",       96, 288),   # 0.33
    ("PEMS04", "./dataset/PEMS/PEMS04.npz",     "pems",       96, 288),   # 0.33
    ("PEMS07", "./dataset/PEMS/PEMS07.npz",     "pems",       96, 288),   # 0.33
    ("PEMS08", "./dataset/PEMS/PEMS08.npz",     "pems",       96, 288),   # 0.33
]
N_WINDOWS = 500
PCA_DIM = 3
PATH_DIM = PCA_DIM + 1
MAX_CHANNELS = 64
EPS = 1e-12
RNG = np.random.default_rng(0)


def load_train(path, kind):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        if "date" in df.columns:
            df = df.drop(columns=["date"])
        x = df.values.astype(np.float64)
    else:
        a = np.load(path, allow_pickle=True)["data"]
        if a.ndim == 3:
            a = a[:, :, 0]
        x = a.astype(np.float64)
    T = len(x)
    if kind == "ett_hour":
        x = x[:12 * 30 * 24]
    elif kind == "ett_minute":
        x = x[:12 * 30 * 24 * 4]
    elif kind == "pems":
        x = x[:int(T * 0.6)]
    if x.shape[1] > MAX_CHANNELS:
        idx = RNG.choice(x.shape[1], size=MAX_CHANNELS, replace=False)
        idx.sort()
        x = x[:, idx]
    return x


def fit_pca3(x):
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + EPS
    xs = (x - mu) / sd
    _, _, Vt = np.linalg.svd(xs, full_matrices=False)
    W = Vt[:PCA_DIM].T
    return mu, sd, W


def time_augment(W):
    Wn, L, D = W.shape
    t = np.linspace(0.0, 1.0, L, dtype=np.float64)
    t = np.broadcast_to(t.reshape(1, L, 1), (Wn, L, 1)).copy()
    return np.concatenate([t, W], axis=-1)


def per_window_level_stats(paths):
    """paths: [W, L, D] -> dict of arrays of shape [W]."""
    Wn, L, D = paths.shape
    S1 = paths[:, -1, :] - paths[:, 0, :]              # [W, D]
    sym_norm = np.zeros(Wn)
    anti_norm = np.zeros(Wn)
    sym_full = np.zeros((Wn, D, D))
    anti_full = np.zeros((Wn, D, D))
    for w in range(Wn):
        sig2 = iisignature.sig(paths[w], 2)            # [D + D^2]
        S2 = sig2[D:].reshape(D, D)
        sym = 0.5 * (S2 + S2.T)
        anti = 0.5 * (S2 - S2.T)
        sym_full[w] = sym
        anti_full[w] = anti
        sym_norm[w] = np.linalg.norm(sym, ord="fro")
        anti_norm[w] = np.linalg.norm(anti, ord="fro")
    s1_norm = np.linalg.norm(S1, axis=1)
    return {
        "S1": S1, "s1_norm": s1_norm,
        "sym_norm": sym_norm, "anti_norm": anti_norm,
        "sym": sym_full, "anti": anti_full,
    }


def stats(arr):
    a = np.asarray(arr, dtype=np.float64)
    m = a.mean()
    sd = a.std()
    cv = sd / (abs(m) + EPS)
    return float(m), float(sd), float(cv), float(np.median(a))


def main():
    out = "figs/sig_levy"
    os.makedirs(out, exist_ok=True)

    bundles = []
    for name, p, kind, L, P in CASES:
        if not os.path.exists(p):
            print(f"missing {p}, skip {name}")
            continue
        train = load_train(p, kind)
        mu, sd, Wmat = fit_pca3(train)
        train_pca = ((train - mu) / sd) @ Wmat                  # [T, 3]
        n = len(train_pca) - L + 1
        starts = RNG.choice(n, size=N_WINDOWS, replace=False)
        wins = np.stack([train_pca[s:s + L] for s in starts])    # [W, L, 3]
        paths = time_augment(wins)                               # [W, L, 4]
        st = per_window_level_stats(paths)
        bundles.append((name, L, P, st))
        print(f"computed {name}  L={L}  P={P}  cycles_per_window={L/P:.2f}")

    rows = []
    for name, L, P, st in bundles:
        m1, s1, cv1, med1 = stats(st["s1_norm"])
        ms, ss, cvs, _ = stats(st["sym_norm"])
        ma, sa, cva, _ = stats(st["anti_norm"])
        ratio = st["anti_norm"] / (st["sym_norm"] + EPS)
        rows.append({
            "dataset": name, "L": L, "P": P,
            "cycles": L / P,
            "mean_S1": m1, "std_S1": s1, "cv_S1": cv1, "median_S1": med1,
            "mean_sym": ms, "cv_sym": cvs,
            "mean_anti": ma, "cv_anti": cva,
            "mean_anti/sym": float(ratio.mean()),
            "frac_anti>sym": float((ratio > 1).mean()),
        })

    df = pd.DataFrame(rows)
    csv = "sig_levy_results.csv"
    df.to_csv(csv, index=False, float_format="%.4f")
    print()
    cols = ["dataset", "L", "P", "cycles", "mean_S1", "cv_S1",
            "mean_sym", "cv_sym", "mean_anti", "cv_anti",
            "mean_anti/sym", "frac_anti>sym"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    colors = {
        "ETTh1": "#1f77b4", "ETTh2": "#4ea3d8",
        "ETTm1": "#ff7f0e", "ETTm2": "#ffb066",
        "PEMS03": "#2ca02c", "PEMS04": "#5cbf5c",
        "PEMS07": "#117a11", "PEMS08": "#8fd58f",
    }

    plt.figure(figsize=(7, 4))
    for name, L, P, st in bundles:
        v = st["s1_norm"]
        plt.hist(v, bins=60, density=True, alpha=0.45, color=colors.get(name),
                 label=f"{name} (L={L}, P={P}, cyc={L/P:.1f})")
    plt.xlabel(r"$\|S^{(1)}\|_2$  (level-1 increment magnitude per window)")
    plt.ylabel("density")
    plt.title("Level-1 norm distribution across windows")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{out}/hist_S1.png", dpi=130); plt.close()

    plt.figure(figsize=(7, 4))
    for name, L, P, st in bundles:
        v = st["anti_norm"]
        plt.hist(np.log10(v + EPS), bins=60, density=True, alpha=0.45,
                 color=colors.get(name),
                 label=f"{name} (cyc={L/P:.1f})")
    plt.xlabel(r"$\log_{10} \|S^{(2)}_{\mathrm{anti}}\|_F$  (Levy area magnitude)")
    plt.ylabel("density")
    plt.title("Antisymmetric level-2 (Levy area) magnitude")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{out}/hist_anti.png", dpi=130); plt.close()

    plt.figure(figsize=(6, 6))
    for name, L, P, st in bundles:
        plt.scatter(st["sym_norm"], st["anti_norm"], s=6, alpha=0.4,
                    color=colors.get(name), label=f"{name}")
    lo = min(b[3]["sym_norm"].min() for b in bundles) + EPS
    hi = max(b[3]["sym_norm"].max() for b in bundles) + EPS
    plt.plot([lo, hi], [lo, hi], "k--", lw=0.5, label="anti = sym")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel(r"$\|S^{(2)}_{\mathrm{sym}}\|_F$ (endpoint-determined)")
    plt.ylabel(r"$\|S^{(2)}_{\mathrm{anti}}\|_F$ (Levy area)")
    plt.title("sym vs anti per window")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{out}/scatter_sym_anti.png", dpi=130); plt.close()

    plt.figure(figsize=(7, 4))
    for name, L, P, st in bundles:
        a = st["anti"]                            # [W, D, D]
        D = a.shape[1]
        labels, vals = [], []
        for i in range(D):
            for j in range(i + 1, D):
                labels.append(f"({i},{j})")
                vals.append(a[:, i, j].std())     # std across windows
        x = np.arange(len(labels))
        plt.plot(x, vals, "-o", color=colors.get(name),
                 label=f"{name} (cyc={L/P:.1f})")
    plt.xticks(x, labels)
    plt.xlabel("channel pair (i, j)  [0=time]")
    plt.ylabel(r"std of Levy area $S^{(2)}_{[i,j]}$ across windows")
    plt.title("Across-window variance of Levy area, per channel pair")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{out}/per_pair_anti_std.png", dpi=130); plt.close()

    print(f"\nsaved: {csv}, plots in {out}/")


if __name__ == "__main__":
    main()
