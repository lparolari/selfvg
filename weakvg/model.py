import pytorch_lightning as pl
import torch
import torch.nn as nn


class WordEmbedding(nn.Module):
    def __init__(self, wordvec, vocab, *, freeze=True):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            wordvec.vectors, freeze=freeze, padding_idx=vocab.get_default_index()
        )

    def forward(self, x):
        return self.embedding(x)


class ConceptModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
