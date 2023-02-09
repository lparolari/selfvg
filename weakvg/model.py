import pytorch_lightning as pl
import torch
import torch.nn as nn

from weakvg.loss import Loss
from weakvg.utils import iou


class MyModel(pl.LightningModule):
    def __init__(self, wordvec, vocab) -> None:
        super().__init__()
        self.we = WordEmbedding(wordvec, vocab)
        # self.linear = torch.nn.Linear(10, 10)
        self.concept = ConceptModel(word_embedding=self.we)
        self.softmax = nn.Softmax(dim=-1)
        self.loss = Loss()

        # self.save_hyperparameters()

    def forward(self, x):
        logits, mask = self.concept(x)

        # -1e8 is required to makes the softmax output 0 as probability for
        # masked values
        logits = logits.masked_fill(~mask, -1e8)  # [b, q, b, p]

        scores = self.softmax(logits)  # [b, q, b, p]

        return scores, mask

    def training_step(self, batch, batch_idx):
        queries = batch["queries"]  # [b, q, w]
        proposals = batch["proposals"]  # [b, p, 4]
        targets = batch["targets"]  # [b, q, 4]

        scores, mask = self.forward(batch)  # [b, q, b, p]

        loss = self.loss(queries, scores)
        loss.requires_grad_()  # TODO: remove this (current workaround for https://discuss.pytorch.org/t/why-do-i-get-loss-does-not-require-grad-and-does-not-have-a-grad-fn/53145)

        candidates = self.predict_candidates(scores, proposals)  # [b, q, 4]

        acc = self.accuracy(
            candidates, targets, queries
        )  # TODO: refactor -> avoid passing queries whenever possible

        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        return batch

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def predict_candidates(self, scores, proposals):
        # scores: [b, q, b, p]
        # proposals: [b, p, 4]

        b = scores.shape[0]
        q = scores.shape[1]
        p = scores.shape[3]

        index = torch.arange(b)  # [b]
        index = index.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [b, 1, 1, 1]
        index = index.repeat(1, q, 1, p)  # [b, q, 1, p]

        scores = scores.gather(-2, index).squeeze(-2)  # [b, q, p]

        predictions = scores.argmax(-1).unsqueeze(-1).repeat(1, 1, 4)  # [b, q, 4]

        candidates = proposals.gather(-2, predictions)  # [b, q, 4]

        return candidates

    def accuracy(self, candidates, targets, queries):
        # scores: [b, q, b, p]
        # candidates: [b, p, 4]
        # targets: [b, q, 4]
        # queries: [b, q, w]

        thresh = 0.5

        scores = iou(candidates, targets)  # [b, q]

        matches = scores >= thresh  # [b, q]

        # mask padding queries
        _, is_query = get_queries_mask(queries)  # [b, q, w], [b, q]
        _, n_queries = get_queries_count(queries)  # [b, q], [b]

        matches = matches.masked_fill(~is_query, False)  # [b, q]

        acc = matches.sum() / n_queries.sum()

        return acc


def get_queries_mask(queries):
    is_word = queries != 0  # [b, q, w]
    is_query = is_word.any(-1)  # [b, q]

    return is_word, is_query


def get_queries_count(queries):
    is_word, is_query = get_queries_mask(queries)

    n_words = is_word.sum(-1)  # [b, q]
    n_queries = is_query.sum(-1)  # [b]

    return n_words, n_queries


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
