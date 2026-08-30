"""Train linear probes to predict 5D moral-framework distributions from activations.

For every transformer layer, trains a linear probe  y_hat = softmax(W x + b)
using KL-divergence loss on soft labels.  Evaluates on a held-out test set,
runs cross-model transfer experiments and a permutation test for statistical
significance, then saves results and probe weights.

Inputs (relative to --data-dir / --results-dir):
    data/probing_dataset.parquet
    data/probing_splits.json
    data/activations_llama.h5
    data/activations_qwen.h5

Outputs (relative to --results-dir):
    probe_results.json
    cross_model_transfer.json
    category_analysis.json
    probe_weights/{model}/layer_{i}.pt
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAMEWORKS = [
    "kantian_deontology",
    "benthamite_act_utilitarianism",
    "aristotelian_virtue_ethics",
    "scanlonian_contractualism",
    "gauthierian_contractarianism",
]
N_CLASSES = len(FRAMEWORKS)
Y_COLS = ["y_kant", "y_util", "y_virtue", "y_contract", "y_contractar"]


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------


class LinearProbe(nn.Module):
    """y_hat = softmax(Wx + b)."""

    def __init__(self, input_dim: int, output_dim: int = N_CLASSES):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.linear(x), dim=-1)


# ---------------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------------


def kl_divergence_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """KL(y_true || y_pred), averaged over the batch."""
    y_pred = torch.clamp(y_pred, eps, 1 - eps)
    y_true = torch.clamp(y_true, eps, 1 - eps)
    return torch.mean(torch.sum(y_true * torch.log(y_true / y_pred), dim=-1))


def evaluate_probe(
    probe: nn.Module,
    X: np.ndarray,
    Y: np.ndarray,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate a trained probe.  Returns (metrics, predictions, kl_values)."""
    probe.eval()
    with torch.no_grad():
        Y_pred = probe(torch.FloatTensor(X).to(device)).cpu().numpy()

    kl_vals = np.array(
        [
            np.sum(Y[i] * np.log((Y[i] + 1e-8) / (Y_pred[i] + 1e-8)))
            for i in range(len(Y))
        ]
    )
    top1_acc = float(np.mean(np.argmax(Y_pred, 1) == np.argmax(Y, 1)))
    pred_ent = scipy_entropy(Y_pred.T + 1e-8)
    true_ent = scipy_entropy(Y.T + 1e-8)
    ent_err = float(np.mean(np.abs(pred_ent - true_ent)))
    per_dim = []
    for d in range(N_CLASSES):
        c = np.corrcoef(Y[:, d], Y_pred[:, d])[0, 1]
        per_dim.append(float(c) if not np.isnan(c) else 0.0)

    metrics = {
        "kl_mean": float(np.mean(kl_vals)),
        "kl_std": float(np.std(kl_vals)),
        "kl_median": float(np.median(kl_vals)),
        "top1_accuracy": top1_acc,
        "entropy_error": ent_err,
        "per_dim_correlation": per_dim,
        "mean_correlation": float(np.mean(per_dim)),
    }
    return metrics, Y_pred, kl_vals


def compute_baselines(Y_train: np.ndarray, Y_test: np.ndarray) -> dict:
    """Uniform and step-prior baselines evaluated on test set."""
    # Uniform
    uniform = np.full_like(Y_test, 0.2)
    kl_u = [
        np.sum(Y_test[i] * np.log((Y_test[i] + 1e-8) / (uniform[i] + 1e-8)))
        for i in range(len(Y_test))
    ]
    # Step-prior
    prior = np.tile(Y_train.mean(axis=0, keepdims=True), (len(Y_test), 1))
    kl_p = [
        np.sum(Y_test[i] * np.log((Y_test[i] + 1e-8) / (prior[i] + 1e-8)))
        for i in range(len(Y_test))
    ]
    return {
        "uniform": {
            "kl_mean": float(np.mean(kl_u)),
            "kl_std": float(np.std(kl_u)),
            "top1_accuracy": float(np.mean(np.argmax(uniform, 1) == np.argmax(Y_test, 1))),
        },
        "step_prior": {
            "kl_mean": float(np.mean(kl_p)),
            "kl_std": float(np.std(kl_p)),
            "top1_accuracy": float(np.mean(np.argmax(prior, 1) == np.argmax(Y_test, 1))),
            "prior_distribution": Y_train.mean(axis=0).tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_probe(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_valid: np.ndarray,
    Y_valid: np.ndarray,
    d_model: int,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    patience: int = 10,
    batch_size: int = 128,
) -> tuple[LinearProbe, dict]:
    """Train a single-layer linear probe with early stopping."""
    Xt = torch.FloatTensor(X_train).to(device)
    Yt = torch.FloatTensor(Y_train).to(device)
    Xv = torch.FloatTensor(X_valid).to(device)
    Yv = torch.FloatTensor(Y_valid).to(device)

    probe = LinearProbe(d_model, N_CLASSES).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)

    history: dict[str, list[float]] = {"train_loss": [], "valid_loss": []}
    best_vloss = float("inf")
    best_weights = None
    wait = 0
    n = Xt.shape[0]

    for _ in range(epochs):
        # Train
        probe.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            loss = kl_divergence_loss(probe(Xt[idx]), Yt[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        history["train_loss"].append(epoch_loss / n)

        # Validate
        probe.eval()
        with torch.no_grad():
            vloss = kl_divergence_loss(probe(Xv), Yv).item()
        history["valid_loss"].append(vloss)

        if vloss < best_vloss:
            best_vloss = vloss
            best_weights = {k: v.clone() for k, v in probe.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_weights is not None:
        probe.load_state_dict(best_weights)
    return probe, history


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_activations(filepath: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Read HDF5 activations file."""
    with h5py.File(filepath, "r") as f:
        act = f["activations"][:]
        sids = [s.decode() for s in f["sample_ids"][:]]
        steps = f["step_ids"][:]
    return act, sids, steps


def build_dataset_arrays(
    model_id: str,
    probing_df: pd.DataFrame,
    activations_data: dict,
    split_scenarios: list[str],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build X (activations), Y (5D distributions), metadata for a split."""
    model_df = probing_df[
        (probing_df["model_id"] == model_id)
        & (probing_df["sample_id"].isin(split_scenarios))
    ]
    ad = activations_data[model_id]
    lookup = {
        (sid, int(step)): i
        for i, (sid, step) in enumerate(zip(ad["sample_ids"], ad["step_ids"]))
    }
    Xs, Ys, metas = [], [], []
    for _, row in model_df.iterrows():
        key = (row["sample_id"], row["step_id"])
        if key in lookup:
            Xs.append(ad["activations"][lookup[key]])
            Ys.append(row[Y_COLS].values.astype(np.float32))
            metas.append(
                {
                    "sample_id": row["sample_id"],
                    "step_id": int(row["step_id"]),
                    "trajectory_category": row["trajectory_category"],
                    "entropy": float(row["entropy"]),
                    "dominant_framework_idx": int(row["dominant_framework_idx"]),
                }
            )
    return np.stack(Xs), np.stack(Ys), metas


# ---------------------------------------------------------------------------
# High-level routines
# ---------------------------------------------------------------------------


def train_all_layer_probes(
    model_id: str,
    datasets: dict,
    device: torch.device,
    save_dir: Path | None = None,
) -> tuple[dict, dict[int, LinearProbe]]:
    """Train probes for every layer; return results dict and probe dict."""
    X_train = datasets[model_id]["train"]["X"]
    Y_train = datasets[model_id]["train"]["Y"]
    X_valid = datasets[model_id]["valid"]["X"]
    Y_valid = datasets[model_id]["valid"]["Y"]
    X_test = datasets[model_id]["test"]["X"]
    Y_test = datasets[model_id]["test"]["Y"]

    n_samples, n_layers, d_model = X_train.shape
    baselines = compute_baselines(Y_train, Y_test)
    print(f"\n{'=' * 60}")
    print(f"Training probes for {model_id}  ({n_layers} layers, d={d_model})")
    print(f"  Uniform baseline KL={baselines['uniform']['kl_mean']:.4f}")
    print(f"  Step-prior baseline KL={baselines['step_prior']['kl_mean']:.4f}")

    results: dict = {
        "model_id": model_id,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_train": len(Y_train),
        "n_valid": len(Y_valid),
        "n_test": len(Y_test),
        "baselines": baselines,
        "layer_results": [],
    }
    probes: dict[int, LinearProbe] = {}
    best_layer, best_kl = -1, float("inf")

    for layer in tqdm(range(n_layers), desc=f"Layers ({model_id})"):
        probe, history = train_probe(
            X_train[:, layer, :], Y_train,
            X_valid[:, layer, :], Y_valid,
            d_model, device,
        )
        probes[layer] = probe
        metrics, _, _ = evaluate_probe(probe, X_test[:, layer, :], Y_test, device)
        results["layer_results"].append(
            {
                "layer": layer,
                "metrics": metrics,
                "final_train_loss": history["train_loss"][-1],
                "final_valid_loss": history["valid_loss"][-1],
                "n_epochs": len(history["train_loss"]),
            }
        )
        if metrics["kl_mean"] < best_kl:
            best_kl = metrics["kl_mean"]
            best_layer = layer

    results["best_layer"] = best_layer
    results["best_kl"] = best_kl
    print(f"  Best layer: {best_layer}  KL={best_kl:.4f}")

    if save_dir is not None:
        wdir = save_dir / "probe_weights" / model_id
        wdir.mkdir(parents=True, exist_ok=True)
        for layer, probe in probes.items():
            torch.save(probe.state_dict(), wdir / f"layer_{layer}.pt")
        print(f"  Weights saved to {wdir}")

    return results, probes


def cross_model_transfer(
    train_model: str,
    test_model: str,
    probes: dict[str, dict[int, LinearProbe]],
    datasets: dict,
    device: torch.device,
) -> dict:
    """Evaluate probes from *train_model* on *test_model*'s test data."""
    print(f"\nTransfer: {train_model} -> {test_model}")
    X_test = datasets[test_model]["test"]["X"]
    Y_test = datasets[test_model]["test"]["Y"]
    n_layers = X_test.shape[1]
    layer_results = []
    for layer in range(n_layers):
        m, _, _ = evaluate_probe(probes[train_model][layer], X_test[:, layer, :], Y_test, device)
        layer_results.append({"layer": layer, "kl_mean": m["kl_mean"], "top1_accuracy": m["top1_accuracy"]})
    best = min(layer_results, key=lambda r: r["kl_mean"])
    print(f"  Best transfer layer: {best['layer']}  KL={best['kl_mean']:.4f}")
    return {
        "train_model": train_model,
        "test_model": test_model,
        "layer_results": layer_results,
        "best_layer": best["layer"],
        "best_kl": best["kl_mean"],
    }


def category_analysis(
    model_id: str,
    best_layer: int,
    probes: dict[int, LinearProbe],
    datasets: dict,
    device: torch.device,
) -> dict:
    """Per-category KL and top-1 accuracy at the best layer."""
    probe = probes[best_layer]
    X = datasets[model_id]["test"]["X"][:, best_layer, :]
    Y = datasets[model_id]["test"]["Y"]
    meta = datasets[model_id]["test"]["metadata"]

    probe.eval()
    with torch.no_grad():
        Yp = probe(torch.FloatTensor(X).to(device)).cpu().numpy()

    groups: dict[str, dict] = {}
    for i, m in enumerate(meta):
        cat = m["trajectory_category"]
        groups.setdefault(cat, {"yt": [], "yp": []})
        groups[cat]["yt"].append(Y[i])
        groups[cat]["yp"].append(Yp[i])

    analysis: dict = {}
    for cat, g in groups.items():
        yt = np.array(g["yt"])
        yp = np.array(g["yp"])
        kl = [np.sum(yt[j] * np.log((yt[j] + 1e-8) / (yp[j] + 1e-8))) for j in range(len(yt))]
        analysis[cat] = {
            "n_samples": len(yt),
            "kl_mean": float(np.mean(kl)),
            "kl_std": float(np.std(kl)),
            "top1_accuracy": float(np.mean(np.argmax(yp, 1) == np.argmax(yt, 1))),
        }
    return analysis


def step_analysis(
    model_id: str,
    best_layer: int,
    probes: dict[int, LinearProbe],
    datasets: dict,
    device: torch.device,
) -> dict:
    """Per-step KL and top-1 accuracy."""
    probe = probes[best_layer]
    X = datasets[model_id]["test"]["X"][:, best_layer, :]
    Y = datasets[model_id]["test"]["Y"]
    meta = datasets[model_id]["test"]["metadata"]

    probe.eval()
    with torch.no_grad():
        Yp = probe(torch.FloatTensor(X).to(device)).cpu().numpy()

    groups: dict[int, list] = {s: [] for s in [1, 2, 3, 4]}
    for i, m in enumerate(meta):
        kl = float(np.sum(Y[i] * np.log((Y[i] + 1e-8) / (Yp[i] + 1e-8))))
        t1 = int(np.argmax(Yp[i]) == np.argmax(Y[i]))
        groups[m["step_id"]].append({"kl": kl, "top1": t1})

    out: dict = {}
    for s in [1, 2, 3, 4]:
        if groups[s]:
            out[s] = {
                "n_samples": len(groups[s]),
                "kl_mean": float(np.mean([r["kl"] for r in groups[s]])),
                "kl_std": float(np.std([r["kl"] for r in groups[s]])),
                "top1_accuracy": float(np.mean([r["top1"] for r in groups[s]])),
            }
    return out


def permutation_test(
    model_id: str,
    best_layer: int,
    datasets: dict,
    device: torch.device,
    n_permutations: int = 100,
) -> dict:
    """Permutation test for statistical significance of probe performance."""
    X_tr = datasets[model_id]["train"]["X"][:, best_layer, :]
    Y_tr = datasets[model_id]["train"]["Y"]
    X_va = datasets[model_id]["valid"]["X"][:, best_layer, :]
    Y_va = datasets[model_id]["valid"]["Y"]
    X_te = datasets[model_id]["test"]["X"][:, best_layer, :]
    Y_te = datasets[model_id]["test"]["Y"]
    d = X_tr.shape[1]

    real_probe, _ = train_probe(X_tr, Y_tr, X_va, Y_va, d, device)
    real_kl = evaluate_probe(real_probe, X_te, Y_te, device)[0]["kl_mean"]
    print(f"\nPermutation test for {model_id} (layer {best_layer}): real KL={real_kl:.4f}")

    perm_kls = []
    for _ in tqdm(range(n_permutations), desc="Permutations"):
        Yp_tr = Y_tr[np.random.permutation(len(Y_tr))]
        Yp_va = Y_va[np.random.permutation(len(Y_va))]
        p, _ = train_probe(X_tr, Yp_tr, X_va, Yp_va, d, device, epochs=50, patience=5)
        perm_kls.append(evaluate_probe(p, X_te, Y_te, device)[0]["kl_mean"])

    perm_kls_arr = np.array(perm_kls)
    p_value = float(np.mean(perm_kls_arr <= real_kl))
    print(f"  Permutation KL: {perm_kls_arr.mean():.4f} +/- {perm_kls_arr.std():.4f}")
    print(f"  p-value: {p_value:.4f}")
    return {
        "real_kl": float(real_kl),
        "permutation_kl_mean": float(perm_kls_arr.mean()),
        "permutation_kl_std": float(perm_kls_arr.std()),
        "p_value": p_value,
        "significant": bool(p_value < 0.05),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train linear probes on hidden-state activations."
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Relative path to data directory (default: data).",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Relative path to results directory (default: results).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device (default: auto-detected).",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=100,
        help="Number of permutation-test iterations (default: 100).",
    )
    parser.add_argument(
        "--skip-transfer", action="store_true",
        help="Skip cross-model transfer experiments.",
    )
    parser.add_argument(
        "--skip-permutation", action="store_true",
        help="Skip permutation test (saves time).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}")

    # 1. Load data ---------------------------------------------------------
    probing_df = pd.read_parquet(data_dir / "probing_dataset.parquet")
    with open(data_dir / "probing_splits.json") as f:
        splits = json.load(f)
    print(f"Loaded probing dataset: {len(probing_df)} instances")

    activations_data: dict = {}
    for mid in ["llama", "qwen"]:
        fp = data_dir / f"activations_{mid}.h5"
        if fp.exists():
            act, sids, steps = load_activations(fp)
            activations_data[mid] = {"activations": act, "sample_ids": sids, "step_ids": steps}
            print(f"  {mid}: {act.shape}")
        else:
            print(f"  {mid}: {fp} not found -- skipping.")

    # 2. Build dataset arrays ----------------------------------------------
    datasets: dict = {}
    for mid in activations_data:
        datasets[mid] = {}
        for split in ["train", "valid", "test"]:
            X, Y, meta = build_dataset_arrays(mid, probing_df, activations_data, splits[mid][split])
            datasets[mid][split] = {"X": X, "Y": Y, "metadata": meta}
        print(f"  {mid} splits: " + ", ".join(
            f"{s}={datasets[mid][s]['X'].shape[0]}" for s in ["train", "valid", "test"]
        ))

    # 3. Train probes for each model --------------------------------------
    all_results: dict = {}
    all_probes: dict[str, dict[int, LinearProbe]] = {}
    for mid in datasets:
        res, probes = train_all_layer_probes(mid, datasets, device, save_dir=results_dir)
        all_results[mid] = res
        all_probes[mid] = probes

    # 4. Cross-model transfer ----------------------------------------------
    transfer_results: dict = {}
    model_ids = list(datasets.keys())
    if not args.skip_transfer and len(model_ids) >= 2:
        transfer_results["llama_to_qwen"] = cross_model_transfer(
            "llama", "qwen", all_probes, datasets, device
        )
        transfer_results["qwen_to_llama"] = cross_model_transfer(
            "qwen", "llama", all_probes, datasets, device
        )

    # 5. Category & step analysis ------------------------------------------
    cat_results: dict = {}
    stp_results: dict = {}
    for mid in all_results:
        bl = all_results[mid]["best_layer"]
        cat_results[mid] = category_analysis(mid, bl, all_probes[mid], datasets, device)
        stp_results[mid] = step_analysis(mid, bl, all_probes[mid], datasets, device)

    # 6. Permutation test --------------------------------------------------
    perm_results: dict = {}
    if not args.skip_permutation:
        for mid in all_results:
            bl = all_results[mid]["best_layer"]
            perm_results[mid] = permutation_test(
                mid, bl, datasets, device, n_permutations=args.n_permutations
            )

    # 7. Save results ------------------------------------------------------
    final = {
        "probe_results": all_results,
        "step_analysis": stp_results,
        "permutation_tests": perm_results,
    }
    with open(results_dir / "probe_results.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved {results_dir / 'probe_results.json'}")

    if transfer_results:
        with open(results_dir / "cross_model_transfer.json", "w") as f:
            json.dump(transfer_results, f, indent=2)
        print(f"Saved {results_dir / 'cross_model_transfer.json'}")

    with open(results_dir / "category_analysis.json", "w") as f:
        json.dump(cat_results, f, indent=2)
    print(f"Saved {results_dir / 'category_analysis.json'}")

    # 8. Summary -----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    for mid, res in all_results.items():
        bl = res["best_layer"]
        bm = res["layer_results"][bl]["metrics"]
        print(f"\n{mid}:")
        print(f"  Best layer: {bl}  KL={res['best_kl']:.4f}")
        print(f"  Top-1 accuracy: {bm['top1_accuracy']:.3f}")
        print(f"  Mean correlation: {bm['mean_correlation']:.3f}")
        if mid in perm_results:
            print(f"  Permutation p-value: {perm_results[mid]['p_value']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
