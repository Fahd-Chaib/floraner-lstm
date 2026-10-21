# FloraNER — Named Entity Recognition with LSTM variants

Lab project — Advanced Natural Language Processing (2026/2027).

We compare a **vanilla LSTM** with two non-vanilla cells, **CIFG** (coupled input-forget gate) and
**LSTMP** (LSTM with projection), on named entity recognition in French botanical text (FloraNER).
All cells are implemented from scratch with `nn.Linear` (no `nn.LSTM`); a **WMC** cell is also included as a bonus.

## Task and data

- **Dataset:** FloraNER (coarse-grained: `ORGANE`, `DESCRIPTEUR`; fine-grained: 10 entity types).
- **Format:** character-level spans are converted to word-level **BIO** tags (`src/data.py`).
- **Preprocessing:** documents (≈254 words on average, up to 1,277) are split into segments at `;` and double spaces (≈14 words on average).
- **Splits:**
  - test = the official test file;
  - validation = 10 % of the training *documents* (fixed seed, identical for all models);
  - one text present in both train and test is removed from train.
- **Metric:** strict **entity-level F1** (same definition as seqeval/conlleval). 79 % of tokens are `O`, so token accuracy is reported but not used for model selection.

## Model

```
Embedding (100) → BiRNN layer(s) built from a custom cell → Dropout → Linear → tag scores
```

| Cell | Equations | Specificity |
|---|---|---|
| Vanilla | (1)–(6) | 4 gates |
| CIFG | (13)–(18) | `i_t = 1 − f_t`, 3 weight blocks (−25 % recurrent params) |
| LSTMP | (19)–(25) | `h_t = W_p m_t + b_p`, recurrent state of size P < H |
| WMC (bonus) | (7)–(12) | gates read the cell state through `tanh(W_c c)` |

`tests/test_cells.py` checks that our vanilla cell matches PyTorch's `nn.LSTM` numerically (max diff ≈ 6e-8), and that padding does not change the outputs, in both directions.

## Repository layout

```
data/{coarse,fine}/{train,test}.csv   FloraNER files (CC BY 4.0)
src/cells.py            Vanilla, CIFG, LSTMP, WMC cells + time loop with padding mask
src/model.py            BiRNN tagger
src/data.py             span → BIO conversion, segmentation, splits, vocabulary
src/metrics.py          entity-level precision / recall / F1
src/train.py            training (early stopping on validation F1), all hyperparameters as CLI flags
src/run_experiments.py  experiment groups (models, matched, lr, optim, init, batch, proj, fine, wmc)
src/aggregate.py        result tables + figures
src/eda.py              exploratory data analysis
run_colab.ipynb         one-click Colab runner
```

## Usage

Single run:

```bash
pip install -r requirements.txt
python tests/test_cells.py
python src/train.py --cell cifg --lr 1e-3 --optimizer adam --batch_size 32 --name my_run
```

A group of experiments, then the result tables and figures:

```bash
python src/run_experiments.py --group models
python src/aggregate.py --results results --out report_assets
```

Every run saves its config, learning curves (`history.csv`), test metrics and the number of trainable parameters in `results/<name>/`.

## Dataset credit

Nainia A., Vignes Lebbe R., Chenin E., Sahraoui M., Mousannif H., Zahir J. (2024).
*FloraNER: a Named Entity Recognition Dataset for Botanical French Text.* Zenodo.
https://doi.org/10.5281/zenodo.12765869 — CC BY 4.0.
