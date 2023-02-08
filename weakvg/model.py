import pytorch_lightning as pl
import torch
import torch.nn as nn

from weakvg.loss import Loss


class MyModel(pl.LightningModule):
    def __init__(self, wordvec, vocab) -> None:
        super().__init__()
        self.we = WordEmbedding(wordvec, vocab)
        # self.linear = torch.nn.Linear(10, 10)
        self.concept = ConceptModel(word_embedding=self.we)
        self.softmax = nn.Softmax(dim=-1)
        self.loss = Loss()

    def forward(self, x):
        logits, mask = self.concept(x)

        # -1e8 is required to makes the softmax output 0 as probability for
        # masked values
        logits = logits.masked_fill(~mask, -1e8)  # [b, q, b, p]

        scores = self.softmax(logits)  # [b, q, b, p]

        return scores, mask

    def training_step(self, batch, batch_idx):
        queries = batch["queries"]  # [b, q, w]

        b = queries.shape[0]

        n_words = (queries != 0).sum(-1)  # [b, q]
        n_queries = n_words.sum(-1).unsqueeze(-1)  # [b, 1]

        scores, mask = self.forward(batch)

        scores, _ = scores.max(-1)  # [b, q, b]

        scores = scores.sum(-2) / n_queries  # [b, b]

        targets = torch.eye(b)  # [b, b]
        targets = targets.argmax(-1)  # [b]

        loss = self.loss(scores, targets)
        loss.requires_grad_()  # TODO: remove this (current workaround for https://discuss.pytorch.org/t/why-do-i-get-loss-does-not-require-grad-and-does-not-have-a-grad-fn/53145)

        return loss

    def validation_step(self, batch, batch_idx):
        return batch

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


class WordEmbedding(nn.Module):
    def __init__(self, wordvec, vocab, *, freeze=True):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            wordvec.vectors, freeze=freeze, padding_idx=vocab.get_default_index()
        )

    def forward(self, x):
        return self.embedding(x)


class ConceptModel(nn.Module):
    def __init__(self, word_embedding):
        super().__init__()
        self.we = word_embedding
        self.sim_fn = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, x):
        queries = x["queries"]  # [b, q, w]
        labels = x["labels"]  # [b, p]

        n_words = (queries != 0).sum(-1).unsqueeze(-1)  # [b, q, 1]

        queries_e = self.we(queries)  # [b, q, w, d]
        queries_e = queries_e.sum(-2) / n_words  # [b, q, d]
        queries_e = queries_e.unsqueeze(-2).unsqueeze(-2)  # [b, q, 1, 1, d]

        labels_e = self.we(labels)  # [b, p, d]
        labels_e = labels_e.unsqueeze(0).unsqueeze(0)  # [1, 1, b, p, d]

        sim = self.sim_fn(queries_e, labels_e)  # [b, q, b, p]

        has_query = (n_words != 0).unsqueeze(-1)  # [b, q, 1, 1]
        has_label = (labels != 0).unsqueeze(0).unsqueeze(0)  # [1, 1, b, p]

        mask = has_query & has_label  # [b, q, b, p]

        return sim, mask
