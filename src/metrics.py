"""F1 au niveau ENTITÉ (même définition que seqeval / conlleval, mode par défaut).

Une entité prédite ne compte comme correcte que si son TYPE et ses LIMITES (début ET fin)
sont exactement ceux de la vraie entité. Ex. : « 15 cm de longueur » prédit « 15 cm »
-> 0 bonne entité, 1 faux positif, 1 faux négatif.
"""
from collections import defaultdict


def get_entities(tags):
    """['B-ORG','I-ORG','O','B-DESC'] -> {('ORG',0,1), ('DESC',3,3)}"""
    ents, start, typ = set(), None, None
    for i, t in enumerate(list(tags) + ["O"]):
        prefix, _, label = t.partition("-")
        # une entité se termine si : O, nouveau B-, ou I- d'un autre type
        if typ is not None and (prefix != "I" or label != typ):
            ents.add((typ, start, i - 1)); typ = None
        # une entité commence sur B-, ou sur un I- isolé (toléré comme dans conlleval)
        if prefix == "B" or (prefix == "I" and typ is None):
            start, typ = i, label
    return ents


def entity_scores(y_true, y_pred):
    """y_true, y_pred : listes de séquences d'étiquettes. Retourne P/R/F1 micro + par type."""
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for k, (t, p) in enumerate(zip(y_true, y_pred)):
        gt, pr = get_entities(t), get_entities(p)
        for typ, *_ in gt & pr: tp[typ] += 1
        for typ, *_ in pr - gt: fp[typ] += 1
        for typ, *_ in gt - pr: fn[typ] += 1

    def prf(a, b, c):
        p = a / (a + b) if a + b else 0.0
        r = a / (a + c) if a + c else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    types = sorted(set(tp) | set(fp) | set(fn))
    per_type = {}
    for typ in types:
        p, r, f = prf(tp[typ], fp[typ], fn[typ])
        per_type[typ] = {"precision": p, "recall": r, "f1": f, "support": tp[typ] + fn[typ]}
    p, r, f = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    macro = sum(v["f1"] for v in per_type.values()) / max(len(per_type), 1)
    return {"precision": p, "recall": r, "f1": f, "macro_f1": macro, "per_type": per_type}
