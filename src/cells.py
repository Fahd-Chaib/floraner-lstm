"""
Cellules LSTM implémentées à la main avec nn.Linear (nn.LSTM interdit).

Astuce d'implémentation : au lieu de 4 nn.Linear séparés pour W_xi, W_xf, W_xo, W_xg,
on utilise UN SEUL nn.Linear(input -> 4*H) qui calcule les 4 produits d'un coup,
puis on découpe le résultat en 4 morceaux (chunk). C'est mathématiquement identique
aux équations de l'annexe (on empile les matrices les unes sous les autres), mais plus rapide sur GPU.
"""
import torch
import torch.nn as nn


class VanillaLSTMCell(nn.Module):
    """Équations (1)-(6) de l'annexe."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = hidden_size          # taille de h (ce qui circule dans la récurrence)
        self.x2g = nn.Linear(input_size, 4 * hidden_size)               # W_x* x_t + b_*
        self.h2g = nn.Linear(hidden_size, 4 * hidden_size, bias=False)  # W_h* h_{t-1}

    def forward(self, x, state):
        h, c = state
        i, f, o, g = (self.x2g(x) + self.h2g(h)).chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)   # (1)(2)(3)
        g = torch.tanh(g)                                                # (4)
        c = f * c + i * g                                                # (5)
        h = o * torch.tanh(c)                                            # (6)
        return h, c


class CIFGCell(nn.Module):
    """Équations (13)-(18) : la porte d'entrée est couplée, i_t = 1 - f_t.
    -> seulement 3 blocs de poids (f, o, g) au lieu de 4 : ~25 % de paramètres récurrents en moins."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = hidden_size
        self.x2g = nn.Linear(input_size, 3 * hidden_size)
        self.h2g = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

    def forward(self, x, state):
        h, c = state
        f, o, g = (self.x2g(x) + self.h2g(h)).chunk(3, dim=-1)
        f = torch.sigmoid(f)                                             # (13)
        i = 1.0 - f                                                      # (14)  <- la seule vraie différence
        o = torch.sigmoid(o)                                             # (15)
        g = torch.tanh(g)                                                # (16)
        c = f * c + i * g                                                # (17)
        h = o * torch.tanh(c)                                            # (18)
        return h, c


class LSTMPCell(nn.Module):
    """Équations (19)-(25) : LSTM avec projection.
    c_t et m_t sont de taille H, mais la sortie h_t est projetée en taille P < H.
    C'est h_t (taille P) qui est réinjecté dans les portes -> les matrices récurrentes
    sont 4*H x P au lieu de 4*H x H."""

    def __init__(self, input_size, hidden_size, proj_size):
        super().__init__()
        assert proj_size < hidden_size, "LSTMP : on veut P < H"
        self.hidden_size = hidden_size
        self.state_size = proj_size            # h_t est de taille P
        self.x2g = nn.Linear(input_size, 4 * hidden_size)
        self.h2g = nn.Linear(proj_size, 4 * hidden_size, bias=False)    # W_h* : H x P
        self.proj = nn.Linear(hidden_size, proj_size)                    # W_p, b_p  (25)

    def forward(self, x, state):
        h, c = state                           # h : (B, P), c : (B, H)
        i, f, o, g = (self.x2g(x) + self.h2g(h)).chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)   # (19)(20)(21)
        g = torch.tanh(g)                                                # (22)
        c = f * c + i * g                                                # (23)
        m = o * torch.tanh(c)                                            # (24)
        h = self.proj(m)                                                 # (25)
        return h, c


class WMCCell(nn.Module):
    """(Bonus) Équations (7)-(12) : Working Memory Connections.
    Les portes lisent aussi la mémoire via tanh(W_c* c). Attention à l'ordre :
    i et f utilisent c_{t-1}, mais o utilise c_t -> on calcule c_t AVANT o_t."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = hidden_size
        self.x2g = nn.Linear(input_size, 4 * hidden_size)
        self.h2g = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.c2if = nn.Linear(hidden_size, 2 * hidden_size, bias=False)  # W_ci, W_cf
        self.c2o = nn.Linear(hidden_size, hidden_size, bias=False)       # W_co

    def forward(self, x, state):
        h, c_prev = state
        i, f, o, g = (self.x2g(x) + self.h2g(h)).chunk(4, dim=-1)
        ci, cf = torch.tanh(self.c2if(c_prev)).chunk(2, dim=-1)
        i = torch.sigmoid(i + ci)                                        # (7)
        f = torch.sigmoid(f + cf)                                        # (8)
        g = torch.tanh(g)                                                # (10)
        c = f * c_prev + i * g                                           # (11)
        o = torch.sigmoid(o + torch.tanh(self.c2o(c)))                   # (9)  utilise c_t
        h = o * torch.tanh(c)                                            # (12)
        return h, c


def make_cell(kind, input_size, hidden_size, proj_size=None):
    if kind == "vanilla":
        return VanillaLSTMCell(input_size, hidden_size)
    if kind == "cifg":
        return CIFGCell(input_size, hidden_size)
    if kind == "lstmp":
        return LSTMPCell(input_size, hidden_size, proj_size)
    if kind == "wmc":
        return WMCCell(input_size, hidden_size)
    raise ValueError(kind)


class RNNLayer(nn.Module):
    """Fait tourner une cellule sur toute la séquence (boucle sur le temps).
    Gère le padding : sur les positions de padding, l'état (h, c) n'est pas mis à jour.
    reverse=True -> lit la phrase de droite à gauche (pour le BiLSTM)."""

    def __init__(self, cell, reverse=False):
        super().__init__()
        self.cell = cell
        self.reverse = reverse

    def forward(self, x, mask):
        # x : (B, T, D), mask : (B, T) avec 1 = vrai mot, 0 = padding
        if self.reverse:
            # le padding est à la fin ; une fois la séquence retournée il est au début,
            # et comme l'état reste à zéro sur le padding, la lecture commence bien au dernier vrai mot.
            x, mask = x.flip(1), mask.flip(1)
        B, T, _ = x.shape
        h = x.new_zeros(B, self.cell.state_size)
        c = x.new_zeros(B, self.cell.hidden_size)
        outs = []
        for t in range(T):
            h_new, c_new = self.cell(x[:, t], (h, c))
            m = mask[:, t].unsqueeze(-1).to(x.dtype)
            h = m * h_new + (1 - m) * h
            c = m * c_new + (1 - m) * c
            outs.append(h * m)
        out = torch.stack(outs, dim=1)
        return out.flip(1) if self.reverse else out
