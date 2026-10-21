"""Modèle NER : Embedding -> BiLSTM (cellules faites main, empilables) -> Dropout -> Linear -> étiquette par mot."""
import torch
import torch.nn as nn
from cells import make_cell, RNNLayer


class BiRNNTagger(nn.Module):
    def __init__(self, vocab_size, num_tags, cell="vanilla", emb_dim=100, hidden=128,
                 proj=None, num_layers=1, bidirectional=True, dropout=0.3, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        in_dim = emb_dim
        for _ in range(num_layers):
            fwd = RNNLayer(make_cell(cell, in_dim, hidden, proj))
            bwd = RNNLayer(make_cell(cell, in_dim, hidden, proj), reverse=True) if bidirectional else None
            self.layers.append(nn.ModuleList([fwd] + ([bwd] if bwd else [])))
            out_dim = fwd.cell.state_size
            in_dim = out_dim * (2 if bidirectional else 1)
        self.out = nn.Linear(in_dim, num_tags)

    def forward(self, tokens, mask):
        x = self.drop(self.emb(tokens))
        for dirs in self.layers:
            x = torch.cat([d(x, mask) for d in dirs], dim=-1)
            x = self.drop(x)
        return self.out(x)          # (B, T, num_tags) : un score par étiquette et par mot


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
