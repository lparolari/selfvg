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
        self.visual_branch = VisualBranch(word_embedding=self.we)
        self.textual_branch = TextualBranch(word_embedding=self.we)
        self.prediction_module = PredictionModule(omega=omega)
        self.loss = Loss()

        self.save_hyperparameters(ignore=["wordvec", "vocab"])

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

        index = torch.arange(b).to(self.device)  # [b]
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
        heads = x["heads"]  # [b, q, h]
        labels = x["labels"]  # [b, p]

        n_heads = (heads != 0).sum(-1).unsqueeze(-1)  # [b, q, 1]

        heads_e = self.we(heads)  # [b, q, h, d]
        heads_e = heads_e.sum(-2) / n_heads  # [b, q, d]
        heads_e = heads_e.unsqueeze(-2).unsqueeze(-2)  # [b, q, 1, 1, d]

        labels_e = self.we(labels)  # [b, p, d]
        labels_e = labels_e.unsqueeze(0).unsqueeze(0)  # [1, 1, b, p, d]

        sim = self.sim_fn(heads_e, labels_e)  # [b, q, b, p]

        has_head = (n_heads != 0).unsqueeze(-1)  # [b, q, 1, 1]
        has_label = (labels != 0).unsqueeze(0).unsqueeze(0)  # [1, 1, b, p]

        mask = has_head & has_label  # [b, q, b, p]

        return sim, mask


class VisualBranch(nn.Module):
    def __init__(self, word_embedding):
        super().__init__()

        self.we = word_embedding

        self.fc = nn.Linear(2048 + 5, 300)
        self.act = nn.LeakyReLU()

        self._init_weights()

    def forward(self, x):
        proposals = x["proposals"]  # [b, p, 4]
        proposals_feat = x["proposals_feat"]  # [b, p, v]
        labels = x["labels"]  # [b, p]

        mask = proposals.greater(0).any(-1).unsqueeze(-1)  # [b, p, 1]

        labels_e = self.we(labels)  # [b, p, d]
        spat = self.spatial(x)  # [b, p, 5]

        proj = self.project(proposals_feat, spat)  # [b, p, d]
        fusion = proj + labels_e  # [b, p, d]

        fusion = fusion.masked_fill(~mask, 0)

        return fusion, mask
    
    def spatial(self, x):
        proposals = x["proposals"]  # [b, p, 4]
        image_w = x["image_w"].unsqueeze(-1)  # [b, 1]
        image_h = x["image_h"].unsqueeze(-1)  # [b, 1]

        x1 = proposals[..., 0] / image_w   # [b, p]
        y1 = proposals[..., 1] / image_h
        x2 = proposals[..., 2] / image_w
        y2 = proposals[..., 3] / image_h

        area = (x2 - x1) * (y2 - y1)  # [b, p]

        spat = torch.stack([x1, y1, x2, y2, area], dim=-1)

        return spat
    
    def project(self, proposals_feat, spat):
        viz = torch.cat([proposals_feat, spat], dim=-1)  # [b, p, v + 5]

        proj = self.fc(viz)  # [b, p, d]
        proj = self.act(proj)

        return proj

    def _init_weights(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)


class TextualBranch(nn.Module):
    def __init__(self, word_embedding):
        super().__init__()
        self.we = word_embedding
        self.lstm = nn.LSTM(300, 300)

    def forward(self, x):
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

        queries = x["queries"]

        b = queries.shape[0]
        q = queries.shape[1]
        w = queries.shape[2]

        _, is_query = get_queries_mask(queries)
        n_words, _ = get_queries_count(queries)

        queries_e = self.we(queries)  # [b, q, w, d]

        d = queries_e.shape[-1]

        queries_e = queries_e.view(-1, w, d)
        queries_e = queries_e.permute(1, 0, 2).contiguous()

        # required on CPU by pytorch, see 
        # https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
        lengths = n_words.view(-1).cpu()
        lengths = lengths.clamp(min=1)  # elements with 0 are not accepted

        queries_packed = pack_padded_sequence(queries_e, lengths, enforce_sorted=False)

        output, hidden = self.lstm(queries_packed)

        queries_x, lengths = pad_packed_sequence(output) 
        # queries_x is a tensor with shape [l, b * q, d], where l is the max length 
        # of the non-padded sequence

        lengths = lengths.to(queries.device)  # back to original device

        # we need to gather the representation of the last word, so we can build an
        # index based on lengths. for example, if we have a query with length 4, its
        # index will be 3

        index = lengths - 1  # [b * q]
        index = index.unsqueeze(0).unsqueeze(-1)  # [1, b * q, 1]
        index = index.repeat(1, 1, d)  # [1, b * q, d]

        queries_x = queries_x.gather(0, index)  # [1, b * q, d]

        queries_x = queries_x.permute(1, 0, 2).contiguous()  # [b * q, 1, d]
        queries_x = queries_x.squeeze(1)  # [b * q, d]
        queries_x = queries_x.view(b, q, d)  # [b, q, d]

        mask = is_query.unsqueeze(-1)

        queries_x  = queries_x.masked_fill(~mask, 0)

        return queries_x, mask
