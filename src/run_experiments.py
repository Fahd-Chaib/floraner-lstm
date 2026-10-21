"""
Plan d'expériences, découpé en GROUPES pour être réparti entre les membres de l'équipe.
Chaque personne lance un groupe sur son propre Colab :

    python src/run_experiments.py --group models
    python src/run_experiments.py --group lr --out_dir /content/drive/MyDrive/rnn_results

Un run déjà terminé (metrics.json présent) est sauté : si Colab se déconnecte,
il suffit de relancer la même commande pour reprendre là où on s'était arrêté.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from train import main as train_main

BASE = []  # hyperparamètres par défaut de train.py : H=128, emb=100, Adam lr=1e-3, batch 32, dropout 0.3

GROUPS = {
    # 1) Comparaison des architectures à taille cachée égale (H=128), 3 graines -> moyenne ± écart-type
    "models": [
        (f"models_{c}_s{s}", ["--cell", c, "--seed", str(s)])
        for c in ["vanilla", "cifg", "lstmp"] for s in [42, 1, 2]
    ],
    # 2) Comparaison à NOMBRE DE PARAMÈTRES égal (~234k params RNN) : est-ce l'architecture
    #    ou juste la taille qui fait la différence ?
    "matched": [
        ("matched_cifg_h154", ["--cell", "cifg", "--hidden", "154"]),
        ("matched_lstmp_h162_p64", ["--cell", "lstmp", "--hidden", "162", "--proj", "64"]),
    ],
    # 3) Learning rate
    "lr": [(f"lr_{lr}", ["--lr", lr]) for lr in ["1e-2", "3e-3", "1e-3", "3e-4"]],
    # 4) Optimiseur (chacun avec un lr adapté à son fonctionnement)
    "optim": [
        ("optim_adam", ["--optimizer", "adam", "--lr", "1e-3"]),
        ("optim_adamw", ["--optimizer", "adamw", "--lr", "1e-3"]),
        ("optim_rmsprop", ["--optimizer", "rmsprop", "--lr", "1e-3"]),
        ("optim_sgd", ["--optimizer", "sgd", "--lr", "0.1"]),
    ],
    # 5) Initialisation des poids
    "init": [
        ("init_default", ["--init", "default"]),
        ("init_xavier", ["--init", "xavier"]),
        ("init_orthogonal", ["--init", "orthogonal"]),
        ("init_orthogonal_fb1", ["--init", "orthogonal", "--forget_bias", "1"]),
    ],
    # 6) Taille de batch
    "batch": [(f"batch_{b}", ["--batch_size", str(b)]) for b in [16, 32, 64, 128]],
    # 7) Spécifique LSTMP : effet de la taille de projection P (H=128 fixé)
    "proj": [(f"proj_p{p}", ["--cell", "lstmp", "--proj", str(p)]) for p in [16, 32, 64, 96]],
    # 8) Bonus : les 3 modèles sur la tâche grain fin (10 types d'entités)
    "fine": [(f"fine_{c}", ["--cell", c, "--granularity", "fine"]) for c in ["vanilla", "cifg", "lstmp"]],
    # 9) Bonus : la 4e cellule (WMC)
    "wmc": [("wmc_base", ["--cell", "wmc"])],
}


def run_group(group, out_dir, cell=None, extra=()):
    for name, args in GROUPS[group]:
        if cell and "--cell" not in args:   # les groupes d'hyperparamètres utilisent vanilla par défaut
            args = ["--cell", cell] + args
            name = f"{name}_{cell}"
        if os.path.exists(os.path.join(out_dir, name, "metrics.json")):
            print(f"[skip] {name} déjà fait"); continue
        print(f"\n===== {name} =====")
        train_main(BASE + list(args) + list(extra) + ["--name", name, "--out_dir", out_dir])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True, choices=list(GROUPS) + ["all"])
    p.add_argument("--out_dir", default="results")
    p.add_argument("--cell", default=None, help="pour refaire un groupe d'hyperparamètres avec une autre cellule")
    a, extra = p.parse_known_args()
    for g in (GROUPS if a.group == "all" else [a.group]):
        run_group(g, a.out_dir, a.cell, extra)
