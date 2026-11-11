# Sequence Models for NLP

**Two end-to-end NLP pipelines in PyTorch. The first is a news classifier built from subword tokens to predictions (BPE/WordPiece → Word2Vec → MLP, 91% accuracy on AG News). The second is a character-level LSTM language model that invents new dinosaur names.**

![Python](https://img.shields.io/badge/python-3.11+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-ee4c2c) ![gensim](https://img.shields.io/badge/gensim-Word2Vec-green) ![HF tokenizers](https://img.shields.io/badge/🤗-tokenizers-yellow)

## Part 1: AG News topic classification

```
raw text ──► BPE / WordPiece tokenizer ──► Word2Vec (Skip-gram vs. CBOW, d ∈ {50,100,200})
                                                   │ mean-pool token vectors
                                                   ▼
                                       MLP ──► {World, Sports, Business, Sci/Tech}
```

| Step | Highlights |
|---|---|
| Tokenization | Trained and compared BPE (GPT-style) and WordPiece (BERT-style) subword tokenizers |
| Embeddings | Trained Skip-gram and CBOW. **Skip-gram gave more coherent neighbours**: 'sport' → equestrian, cycling, rowing, olympics |
| Classifier | MLP over mean-pooled embeddings |
| Evaluation | **90.95% test accuracy** on 7,600 samples. The confusion matrix shows most errors fall between Business and Sci/Tech, the two topics with the most overlap |

## Part 2: Char-RNN dinosaur-name generator

- Custom PyTorch `Dataset` with `<` and `>` start/end tokens and a character vocabulary, plus train/validation splits and dataloaders.
- One-hot inputs feed a 2-layer LSTM (256 hidden units, dropout 0.5) with a linear head. A layer-by-layer shape walkthrough of the forward pass is included.
- Generation strategies: random sampling, top-k, and temperature.

Sample generated names: `<rexachisaurus>`, `<eurodon>`, `<wellmertosaurus>`, `<dorghisaurus>`

**Finding:** top-k had little visible effect at this vocabulary size (28 characters). Temperature is the real dial. At low temperature the model produces safe, familiar names. At T=1.5 it invents names until they become gibberish (`nappodromeevoagduaut`).

## Run it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/text_classification_and_char_rnn.ipynb
```

Developed on Google Colab. A GPU is recommended for the LSTM.

## Context

This was homework 2 of the **LLM Architectures** module in **Nebius Academy: AI Performance Engineering**. The task formulations are kept inline in the notebook.

## Authors

**Nathanael Bloch** · **Tomer Porat**
