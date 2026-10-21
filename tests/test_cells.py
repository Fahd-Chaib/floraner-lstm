import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, torch.nn as nn
from cells import VanillaLSTMCell, RNNLayer, make_cell
from model import BiRNNTagger, count_params
torch.manual_seed(0)
D, H, B, T = 8, 16, 3, 7

# 1) Vanilla == nn.LSTM officiel (référence uniquement pour le test)
ref = nn.LSTM(D, H, batch_first=True)
cell = VanillaLSTMCell(D, H)
def reorder(w):  # nn.LSTM range les portes (i,f,g,o), nous (i,f,o,g)
    i, f, g, o = w.chunk(4, 0); return torch.cat([i, f, o, g], 0)
with torch.no_grad():
    cell.x2g.weight.copy_(reorder(ref.weight_ih_l0))
    cell.x2g.bias.copy_(reorder(ref.bias_ih_l0 + ref.bias_hh_l0))
    cell.h2g.weight.copy_(reorder(ref.weight_hh_l0))
x = torch.randn(B, T, D); mask = torch.ones(B, T)
diff = (RNNLayer(cell)(x, mask) - ref(x)[0]).abs().max().item()
print(f"1) Vanilla vs nn.LSTM : écart max = {diff:.2e}", "OK" if diff < 1e-5 else "ERREUR")

# 2) Le padding ne change rien (avant et arrière)
for rev in [False, True]:
    for kind in ["vanilla", "cifg", "lstmp", "wmc"]:
        layer = RNNLayer(make_cell(kind, D, H, 8), reverse=rev)
        x1 = torch.randn(1, 5, D)
        xp = torch.cat([x1, torch.randn(1, 3, D)], 1)
        mp = torch.tensor([[1,1,1,1,1,0,0,0.]])
        d = (layer(x1, torch.ones(1,5)) - layer(xp, mp)[:, :5]).abs().max().item()
        assert d < 1e-6, (kind, rev, d)
print("2) Padding sans effet sur les 4 cellules, dans les 2 sens : OK")

# 3) Nombre de paramètres de chaque cellule seule (D=100, H=128, P=64)
for kind in ["vanilla", "cifg", "lstmp", "wmc"]:
    print(f"3) {kind:8s} cellule : {count_params(make_cell(kind, 100, 128, 64)):,} paramètres")

# 4) Le modèle complet tourne et la loss se calcule
m = BiRNNTagger(vocab_size=500, num_tags=5, cell="lstmp", proj=64)
tok = torch.randint(1, 500, (B, T)); out = m(tok, mask)
print("4) Sortie du modèle :", tuple(out.shape), "= (batch, mots, étiquettes) OK")
