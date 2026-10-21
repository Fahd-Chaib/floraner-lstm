"""
Préparation de FloraNER pour un modèle de NER au niveau mot.

Étapes :
1. Les annotations sont des positions de caractères (début, fin, TYPE) -> on découpe le texte
   en mots et on donne à chaque mot une étiquette BIO (B-TYPE, I-TYPE ou O).
2. Les descriptions sont longues (jusqu'à ~1300 mots). On les coupe en SEGMENTS aux « ; »
   et aux doubles espaces, qui séparent les descriptions d'organes dans les flores.
   Les segments trop longs sont recoupés à MAX_LEN mots.
3. Validation : 10 % des DOCUMENTS d'entraînement (et non des segments, pour éviter que deux
   morceaux du même texte se retrouvent dans train et validation). Graine fixe -> mêmes splits
   pour tous les modèles.
4. Vocabulaire construit sur le train uniquement ; chiffres normalisés (« 12 » et « 35 » -> « 00 ») ;
   mots vus une seule fois -> <unk>, pour que le modèle apprenne quoi faire d'un mot inconnu.
"""
import ast
import random
import re
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset

TOKEN_RE = re.compile(r"\w+(?:-\w+)*|[^\w\s]")
PAD, UNK = "<pad>", "<unk>"
MAX_LEN = 100


def tokenize(text):
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def to_bio(text, spans):
    """Retourne la liste des (mot, début, fin) et la liste des étiquettes BIO."""
    toks = tokenize(text)
    tags = ["O"] * len(toks)
    for s, e, lab in sorted(spans):
        idx = [i for i, (_, ts, te) in enumerate(toks) if ts < e and te > s]
        for k, i in enumerate(idx):
            tags[i] = ("B-" if k == 0 else "I-") + lab
    return toks, tags


def segment(toks, tags, max_len=MAX_LEN):
    """Coupe un document en segments aux « ; » et aux doubles espaces."""
    segs, cur_w, cur_t = [], [], []
    for k, ((w, s, e), t) in enumerate(zip(toks, tags)):
        cur_w.append(w); cur_t.append(t)
        nxt_start = toks[k + 1][1] if k + 1 < len(toks) else None
        boundary = w == ";" or (nxt_start is not None and nxt_start - e >= 2)
        if boundary or len(cur_w) >= max_len:
            # on évite de couper au milieu d'une entité (I- au début du segment suivant)
            if not boundary and k + 1 < len(tags) and tags[k + 1].startswith("I-"):
                continue
            segs.append((cur_w, cur_t)); cur_w, cur_t = [], []
    if cur_w:
        segs.append((cur_w, cur_t))
    # un segment qui ne contient que de la ponctuation n'apporte rien
    return [(w, t) for w, t in segs if any(x.isalnum() for x in "".join(w))]


def read_docs(path):
    df = pd.read_csv(path)
    return [(text, ast.literal_eval(ann)) for text, ann in zip(df.Text, df.filtered_annotations)]


def docs_to_segments(docs):
    out = []
    for text, spans in docs:
        toks, tags = to_bio(text, spans)
        out.extend(segment(toks, tags))
    return out


def load_splits(data_dir, granularity="coarse", val_ratio=0.1, seed=42):
    train_docs = read_docs(f"{data_dir}/{granularity}/train.csv")
    test_docs = read_docs(f"{data_dir}/{granularity}/test.csv")
    # retire les textes présents à la fois dans train et test (fuite de données)
    test_texts = {t for t, _ in test_docs}
    train_docs = [d for d in train_docs if d[0] not in test_texts]
    rng = random.Random(seed)
    rng.shuffle(train_docs)
    n_val = int(len(train_docs) * val_ratio)
    val_docs, train_docs = train_docs[:n_val], train_docs[n_val:]
    return {
        "train": docs_to_segments(train_docs),
        "val": docs_to_segments(val_docs),
        "test": docs_to_segments(test_docs),
        "n_docs": {"train": len(train_docs), "val": len(val_docs), "test": len(test_docs)},
    }


def normalize(word):
    return re.sub(r"\d", "0", word.lower())


class Vocab:
    def __init__(self, train_segments, min_freq=2):
        counts = Counter(normalize(w) for ws, _ in train_segments for w in ws)
        self.itos = [PAD, UNK] + sorted(w for w, c in counts.items() if c >= min_freq)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        tags = sorted({t for _, ts in train_segments for t in ts} - {"O"})
        self.tags = ["O"] + tags
        self.tag2id = {t: i for i, t in enumerate(self.tags)}

    def encode_words(self, words):
        return [self.stoi.get(normalize(w), 1) for w in words]

    def encode_tags(self, tags):
        return [self.tag2id.get(t, 0) for t in tags]


class NERDataset(Dataset):
    def __init__(self, segments, vocab):
        self.items = [(vocab.encode_words(w), vocab.encode_tags(t)) for w, t in segments]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


IGNORE = -100  # étiquette des positions de padding : ignorée par la loss


def collate(batch):
    T = max(len(w) for w, _ in batch)
    tokens = torch.zeros(len(batch), T, dtype=torch.long)
    tags = torch.full((len(batch), T), IGNORE, dtype=torch.long)
    mask = torch.zeros(len(batch), T)
    for b, (w, t) in enumerate(batch):
        tokens[b, :len(w)] = torch.tensor(w)
        tags[b, :len(t)] = torch.tensor(t)
        mask[b, :len(w)] = 1
    return tokens, tags, mask
