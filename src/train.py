"""
Entraîne un modèle NER et enregistre ses résultats.

Exemple :
    python src/train.py --cell cifg --hidden 128 --lr 1e-3 --optimizer adam --name cifg_base

Chaque run écrit dans results/<name>/ :
    config.json   -> tous les hyperparamètres
    history.csv   -> loss et F1 à chaque époque (courbes d'apprentissage)
    metrics.json  -> F1 test final, nombre de paramètres, temps d'entraînement
    best.pt       -> poids du meilleur modèle (meilleur F1 validation)
"""
import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data import IGNORE, NERDataset, Vocab, collate, load_splits
from metrics import entity_scores
from model import BiRNNTagger, count_params


def get_args(argv=None):
    p = argparse.ArgumentParser()
    # données
    p.add_argument("--data_dir", default="data")
    p.add_argument("--granularity", default="coarse", choices=["coarse", "fine"])
    # architecture
    p.add_argument("--cell", default="vanilla", choices=["vanilla", "cifg", "lstmp", "wmc"])
    p.add_argument("--emb_dim", type=int, default=100)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--proj", type=int, default=64, help="taille P pour LSTMP")
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--unidirectional", action="store_true")
    p.add_argument("--dropout", type=float, default=0.3)
    # optimisation
    p.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "sgd", "rmsprop"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5, help="early stopping sur le F1 validation")
    p.add_argument("--clip", type=float, default=5.0, help="gradient clipping (norme max)")
    p.add_argument("--init", default="default", choices=["default", "xavier", "orthogonal"])
    p.add_argument("--forget_bias", type=float, default=0.0, help="biais initial de la porte d'oubli")
    # divers
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default=None)
    p.add_argument("--out_dir", default="results")
    p.add_argument("--max_batches", type=int, default=None, help="debug : limite le nb de batchs")
    return p.parse_args(argv)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def init_weights(model, scheme, forget_bias):
    """default = init uniforme de PyTorch ; xavier = Glorot ; orthogonal = matrices récurrentes orthogonales."""
    for name, p in model.named_parameters():
        if "emb" in name:
            continue
        if scheme != "default" and p.dim() == 2:
            if scheme == "orthogonal" and "h2g" in name:
                pass  # traité plus bas, porte par porte
            else:
                nn.init.xavier_uniform_(p)
        elif scheme != "default" and p.dim() == 1:
            nn.init.zeros_(p)
    if scheme == "orthogonal":
        # une matrice orthogonale par porte : W_hi, W_hf, ... (blocs de H lignes)
        for layer in model.layers:
            for d in layer:
                W = d.cell.h2g.weight.data
                for blk in W.split(d.cell.hidden_size, 0):
                    nn.init.orthogonal_(blk)
    if forget_bias:
        # la porte f est le 2e bloc (vanilla/lstmp/wmc : i,f,o,g) ou le 1er (cifg : f,o,g)
        for layer in model.layers:
            for d in layer:
                cell = d.cell
                H = cell.hidden_size
                start = 0 if type(cell).__name__ == "CIFGCell" else H
                with torch.no_grad():
                    cell.x2g.bias[start:start + H].fill_(forget_bias)


def make_optimizer(name, params, lr):
    if name == "adam": return torch.optim.Adam(params, lr=lr)
    if name == "adamw": return torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    if name == "sgd": return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "rmsprop": return torch.optim.RMSprop(params, lr=lr)


@torch.no_grad()
def evaluate(model, loader, vocab, device, loss_fn):
    model.eval()
    y_true, y_pred, tot_loss, n = [], [], 0.0, 0
    for tokens, tags, mask in loader:
        tokens, tags, mask = tokens.to(device), tags.to(device), mask.to(device)
        logits = model(tokens, mask)
        tot_loss += loss_fn(logits.reshape(-1, logits.size(-1)), tags.reshape(-1)).item(); n += 1
        pred = logits.argmax(-1)
        for b in range(tokens.size(0)):
            L = int(mask[b].sum())
            y_true.append([vocab.tags[i] for i in tags[b, :L].tolist()])
            y_pred.append([vocab.tags[i] for i in pred[b, :L].tolist()])
    scores = entity_scores(y_true, y_pred)
    flat_t = [t for s in y_true for t in s]; flat_p = [t for s in y_pred for t in s]
    scores["token_accuracy"] = float(np.mean([a == b for a, b in zip(flat_t, flat_p)]))
    scores["loss"] = tot_loss / max(n, 1)
    return scores, y_true, y_pred


def main(argv=None):
    args = get_args(argv)
    name = args.name or f"{args.cell}_{args.granularity}"
    out = os.path.join(args.out_dir, name)
    os.makedirs(out, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    S = load_splits(args.data_dir, args.granularity, seed=42)   # splits toujours identiques
    vocab = Vocab(S["train"])
    loaders = {
        k: DataLoader(NERDataset(S[k], vocab), batch_size=args.batch_size,
                      shuffle=(k == "train"), collate_fn=collate)
        for k in ["train", "val", "test"]
    }

    model = BiRNNTagger(len(vocab.itos), len(vocab.tags), cell=args.cell, emb_dim=args.emb_dim,
                        hidden=args.hidden, proj=args.proj if args.cell == "lstmp" else None,
                        num_layers=args.layers, bidirectional=not args.unidirectional,
                        dropout=args.dropout).to(device)
    init_weights(model, args.init, args.forget_bias)
    n_params = count_params(model)
    n_rnn_params = sum(p.numel() for n_, p in model.named_parameters() if n_.startswith("layers"))
    print(f"[{name}] {args.cell} | params total {n_params:,} | params RNN {n_rnn_params:,} | device {device}")

    opt = make_optimizer(args.optimizer, model.parameters(), args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE)
    json.dump(vars(args), open(os.path.join(out, "config.json"), "w"), indent=2)

    best_f1, best_epoch, bad, history = -1, 0, 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); tl, nb = 0.0, 0
        for bi, (tokens, tags, mask) in enumerate(loaders["train"]):
            if args.max_batches and bi >= args.max_batches: break
            tokens, tags, mask = tokens.to(device), tags.to(device), mask.to(device)
            logits = model(tokens, mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tags.reshape(-1))
            opt.zero_grad(); loss.backward()
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step(); tl += loss.item(); nb += 1
            if not torch.isfinite(loss): print("loss NaN -> arrêt"); break
        val, _, _ = evaluate(model, loaders["val"], vocab, device, loss_fn)
        row = {"epoch": epoch, "train_loss": tl / max(nb, 1), "val_loss": val["loss"],
               "val_f1": val["f1"], "val_acc": val["token_accuracy"], "time": time.time() - t0}
        history.append(row)
        print(f"  ep {epoch:2d} | train loss {row['train_loss']:.4f} | val loss {val['loss']:.4f} "
              f"| val F1 {val['f1']:.4f} | {row['time']:.0f}s")
        if val["f1"] > best_f1:
            best_f1, best_epoch, bad = val["f1"], epoch, 0
            torch.save(model.state_dict(), os.path.join(out, "best.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stopping (pas d'amélioration depuis {args.patience} époques)"); break
    train_time = time.time() - t0

    with open(os.path.join(out, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys()); w.writeheader(); w.writerows(history)

    model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=device))
    test, y_true, y_pred = evaluate(model, loaders["test"], vocab, device, loss_fn)
    try:  # vérification croisée avec seqeval quand il est installé (ex. sur Colab)
        from seqeval.metrics import f1_score
        test["seqeval_f1"] = f1_score(y_true, y_pred)
    except ImportError:
        pass
    res = {"name": name, "cell": args.cell, "granularity": args.granularity,
           "params_total": n_params, "params_rnn": n_rnn_params,
           "best_epoch": best_epoch, "val_f1": best_f1, "test_f1": test["f1"],
           "test_precision": test["precision"], "test_recall": test["recall"],
           "test_macro_f1": test["macro_f1"], "test_token_acc": test["token_accuracy"],
           "test_per_type": test["per_type"], "seqeval_f1": test.get("seqeval_f1"),
           "train_time_s": train_time, "epochs_run": len(history),
           "sec_per_epoch": train_time / len(history), "config": vars(args)}
    json.dump(res, open(os.path.join(out, "metrics.json"), "w"), indent=2, ensure_ascii=False)
    print(f"[{name}] TEST F1 = {test['f1']:.4f} (P {test['precision']:.3f} / R {test['recall']:.3f}) "
          f"| acc tokens {test['token_accuracy']:.3f}")
    return res


if __name__ == "__main__":
    main()
