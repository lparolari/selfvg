import pytorch_lightning as pl
import torch
import torch.nn as nn

from weakvg.loss import Loss
from weakvg.utils import iou


class MyModel(pl.LightningModule):
    def __init__(self, wordvec, vocab, omega=0.5) -> None:
        super().__init__()
        self.we = WordEmbedding(wordvec, vocab)
        self.concept_branch = ConceptBranch(word_embedding=self.we)
        self.visual_branch = VisualBranch()
        self.textual_branch = TextualBranch()
        self.prediction_module = PredictionModule(omega=omega)
        self.loss = Loss()

        # self.save_hyperparameters()

    def forward(self, x):
        queries = x["queries"]
        proposals = x["proposals"]

        b = queries.shape[0]
        q = queries.shape[1]
        p = proposals.shape[1]

        concepts_pred, concepts_mask = self.concept_branch(x)  # [b, q, b, p]

        viz, viz_mask = self.visual_branch(x)  # [b, p, d], [b, p, 1]
        viz = viz.unsqueeze(0).unsqueeze(0).repeat(b, q, 1, 1, 1)  # [b, q, b, p, d]
        viz_mask = (
            viz_mask.unsqueeze(0).unsqueeze(0).squeeze(-1).repeat(b, q, 1, 1)
        )  # [b, q, b, p]

        tex, tex_mask = self.textual_branch(x)  # [b, q, d], [b, q, 1]
        tex = tex.unsqueeze(2).unsqueeze(2).repeat(1, 1, b, p, 1)  # [b, q, b, p, d]
        tex_mask = tex_mask.unsqueeze(-1).repeat(1, 1, b, p)  # [b, q, b, p]

        network_pred = torch.cosine_similarity(viz, tex, dim=-1)  # [b, q, b, p]
        network_mask = viz_mask & tex_mask  # [b, q, b, p]

        return self.prediction_module(
            {
                "network_pred": network_pred,
                "network_mask": network_mask,
                "concepts_pred": concepts_pred,
                "concepts_mask": concepts_mask,
            }
        )

    def training_step(self, batch, batch_idx):
        queries = batch["queries"]  # [b, q, w]
        proposals = batch["proposals"]  # [b, p, 4]
        targets = batch["targets"]  # [b, q, 4]

        scores, mask = self.forward(batch)  # [b, q, b, p]

        loss = self.loss(queries, scores)

        candidates = self.predict_candidates(scores, proposals)  # [b, q, 4]

        acc = self.accuracy(
            candidates, targets, queries
        )  # TODO: refactor -> avoid passing queries whenever possible

        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        queries = batch["queries"]  # [b, q, w]
        proposals = batch["proposals"]  # [b, p, 4]
        targets = batch["targets"]  # [b, q, 4]

        scores, mask = self.forward(batch)  # [b, q, b, p]

        loss = self.loss(queries, scores)

        candidates = self.predict_candidates(scores, proposals)  # [b, q, 4]

        acc = self.accuracy(candidates, targets, queries)

        self.log("val_loss", loss)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)

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


class PredictionModule(nn.Module):
    def __init__(self, omega):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.omega = omega

    def forward(self, x):
        concepts_pred = x["concepts_pred"]  # [b, q, b, p]
        concepts_mask = x["concepts_mask"]  # [b, q, b, p]
        network_pred = x["network_pred"]  # [b, q, b, p]
        network_mask = x["network_mask"]  # [b, q, b, p]

        # -1e8 is required to makes the softmax output 0 as probability for
        # masked values
        concepts_pred = concepts_pred.masked_fill(~concepts_mask, -1e8)  # [b, q, b, p]
        network_pred = network_pred.masked_fill(~network_mask, -1e8)  # [b, p, b, p]

        network_contrib = self.omega * self.softmax(network_pred)  # [b, q, b, p]
        concept_contrib = (1 - self.omega) * self.softmax(concepts_pred)  # [b, q, b, p]

        scores = network_contrib + concept_contrib  # [b, q, b, p]

        return scores, concepts_mask


class WordEmbedding(nn.Module):
    def __init__(self, wordvec, vocab, *, freeze=True):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            wordvec.vectors, freeze=freeze, padding_idx=vocab.get_default_index()
        )

    def forward(self, x):
        return self.embedding(x)


class ConceptBranch(nn.Module):
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


class VisualBranch(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(2048, 300)
        self.act = nn.LeakyReLU()

        self._init_weights()

    def forward(self, x):
        proposals = x["proposals"]  # [b, p, 4]
        proposals_feat = x["proposals_feat"]  # [b, p, v]

        mask = proposals.greater(0).any(-1).unsqueeze(-1)  # [b, p, 1]

        fusion = self.fc(proposals_feat)
        fusion = self.act(fusion)
        fusion = fusion.masked_fill(~mask, 0)

        return fusion, mask

    def _init_weights(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)


class TextualBranch(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        queries = x["queries"]

        b = queries.shape[0]
        q = queries.shape[1]

        _, is_query = get_queries_mask(queries)

        mask = is_query.unsqueeze(-1)  # TODO: temp

        return torch.zeros(b, q, 300), mask
