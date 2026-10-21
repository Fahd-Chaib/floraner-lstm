"""
Rassemble tous les résultats (results/*/metrics.json) en tableaux + figures pour le rapport.

    python src/aggregate.py --results results --out report_assets
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load(results_dir):
    rows = []
    for f in glob.glob(os.path.join(results_dir, "*", "metrics.json")):
        m = json.load(open(f))
        c = m["config"]
        rows.append({
            "name": m["name"], "group": m["name"].split("_")[0], "cell": m["cell"],
            "granularity": m["granularity"], "hidden": c["hidden"],
            "proj": c["proj"] if m["cell"] == "lstmp" else None,
            "optimizer": c["optimizer"], "lr": c["lr"], "batch": c["batch_size"],
            "init": c["init"], "forget_bias": c["forget_bias"], "seed": c["seed"],
            "params_rnn": m["params_rnn"], "params_total": m["params_total"],
            "val_f1": m["val_f1"], "test_f1": m["test_f1"], "test_P": m["test_precision"],
            "test_R": m["test_recall"], "test_macro_f1": m["test_macro_f1"],
            "test_token_acc": m["test_token_acc"], "best_epoch": m["best_epoch"],
            "sec_per_epoch": m["sec_per_epoch"],
        })
    return pd.DataFrame(rows).sort_values("name")


def curves(results_dir, names, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for n in names:
        h = os.path.join(results_dir, n, "history.csv")
        if not os.path.exists(h): continue
        d = pd.read_csv(h)
        axes[0].plot(d.epoch, d.val_loss, label=n)
        axes[1].plot(d.epoch, d.val_f1, label=n)
    axes[0].set(title="Validation loss", xlabel="epoch")
    axes[1].set(title="Validation entity F1", xlabel="epoch")
    axes[1].legend(fontsize=7)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="report_assets")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load(a.results)
    if df.empty:
        print("Aucun résultat trouvé."); return
    df.to_csv(os.path.join(a.out, "all_runs.csv"), index=False)
    cols = ["name", "params_rnn", "val_f1", "test_f1", "test_P", "test_R", "test_token_acc", "best_epoch", "sec_per_epoch"]

    # Tableau principal : moyenne ± écart-type sur les graines
    m = df[df.group == "models"]
    if not m.empty:
        t = m.groupby("cell").agg(params_rnn=("params_rnn", "first"),
                                  test_f1_mean=("test_f1", "mean"), test_f1_std=("test_f1", "std"),
                                  test_P=("test_P", "mean"), test_R=("test_R", "mean"),
                                  sec_per_epoch=("sec_per_epoch", "mean"), n_seeds=("seed", "count"))
        t.to_csv(os.path.join(a.out, "table_models.csv"))
        print("\n=== Comparaison des architectures (H=128, moyenne sur les graines) ===")
        print(t.round(4).to_string())
        # F1 vs nombre de paramètres
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for cell, g in m.groupby("cell"):
            ax.errorbar(g.params_rnn.mean(), g.test_f1.mean(), yerr=g.test_f1.std(), fmt="o", capsize=4, label=cell)
        mm = df[df.group == "matched"]
        for _, r in mm.iterrows():
            ax.scatter(r.params_rnn, r.test_f1, marker="x", label=r["name"])
        ax.set(xlabel="RNN trainable parameters", ylabel="Test entity F1", title="F1 vs parameters")
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(a.out, "f1_vs_params.png"), dpi=150)
        plt.close(fig)
        curves(a.results, [n for n in m.name if n.endswith("_s42")], "Architectures (seed 42)",
               os.path.join(a.out, "curves_models.png"))

    for g in ["matched", "lr", "optim", "init", "batch", "proj", "fine", "wmc"]:
        d = df[df.group == g]
        if d.empty: continue
        print(f"\n=== {g} ===")
        print(d[cols].round(4).to_string(index=False))
        d[cols].to_csv(os.path.join(a.out, f"table_{g}.csv"), index=False)
        curves(a.results, list(d.name), g, os.path.join(a.out, f"curves_{g}.png"))
    print(f"\nTableaux et figures dans {a.out}/")


if __name__ == "__main__":
    main()
