"""
Analyse exploratoire (EDA) : statistiques + figures pour le rapport et les slides.

    python src/eda.py --out report_assets/eda
"""
import argparse
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data import Vocab, load_splits, normalize, read_docs, tokenize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out", default="report_assets/eda")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lines = []
    for g in ["coarse", "fine"]:
        docs = read_docs(f"{a.data_dir}/{g}/train.csv")
        doc_len = [len(tokenize(t)) for t, _ in docs]
        S = load_splits(a.data_dir, g)
        v = Vocab(S["train"])
        tags = Counter(t for _, ts in S["train"] for t in ts)
        ents = Counter(t[2:] for t in tags.elements() if t.startswith("B-"))
        seg_len = [len(w) for w, _ in S["train"]]
        test_words = [normalize(w) for ws, _ in S["test"] for w in ws]
        unk = np.mean([w not in v.stoi for w in test_words])
        lines += [f"## {g}",
                  f"documents train/val/test : {S['n_docs']}",
                  f"segments train/val/test : {len(S['train'])}/{len(S['val'])}/{len(S['test'])}",
                  f"mots par document : moyenne {np.mean(doc_len):.0f}, médiane {np.median(doc_len):.0f}, max {max(doc_len)}",
                  f"mots par segment : moyenne {np.mean(seg_len):.1f}, médiane {np.median(seg_len):.0f}, p95 {np.percentile(seg_len, 95):.0f}",
                  f"part des mots 'O' : {100 * tags['O'] / sum(tags.values()):.1f} %",
                  f"vocabulaire (min_freq=2) : {len(v.itos)} | mots du test inconnus : {100 * unk:.1f} %",
                  f"entités (train) : {dict(ents.most_common())}", ""]

        fig, ax = plt.subplots(figsize=(6, 3.5))
        k, c = zip(*ents.most_common())
        ax.barh(k[::-1], c[::-1]); ax.set_xscale("log")
        ax.set(title=f"Entity counts ({g}, train, log scale)", xlabel="count")
        fig.tight_layout(); fig.savefig(f"{a.out}/entities_{g}.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(doc_len, bins=50); axes[0].set(title="Words per document", xlabel="words")
    axes[1].hist(seg_len, bins=50); axes[1].set(title="Words per segment (after splitting)", xlabel="words")
    fig.tight_layout(); fig.savefig(f"{a.out}/lengths.png", dpi=150); plt.close(fig)

    txt = "\n".join(lines)
    open(f"{a.out}/stats.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
