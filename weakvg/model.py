import pytorch_lightning as pl
import torch
import torch.nn as nn

from weakvg.loss import Loss
from weakvg.utils import (
    ext_textual,
    ext_visual,
    get_queries_count,
    get_queries_mask,
    iou,
    mask_softmax,
    tlbr2ctwh,
)


class MyModel(pl.LightningModule):
    def __init__(
        self,
        wordvec,
        vocab,
        omega=0.5,
        neg_selection="random",
        grounding="similarity",
    ) -> None:
        super().__init__()
        self.we = WordEmbedding(wordvec, vocab)
        self.concept_branch = ConceptBranch(word_embedding=self.we)
        self.visual_branch = VisualBranch(word_embedding=self.we)
        self.textual_branch = TextualBranch(word_embedding=self.we)
    
        if grounding == "similarity":
            self.prediction_module = SimilarityPredictionModule(omega=omega)
        if grounding == "nn":
            self.prediction_module = NeuralNetworkPredictionModule(omega=omega)

        self.loss = Loss(word_embedding=self.we, neg_selection=neg_selection)

        self.save_hyperparameters(ignore=["wordvec", "vocab"])

    def forward(self, x):
        concepts_pred, concepts_mask = self.concept_branch(x)  # [b, q, b, p]

        visual_feat, visual_mask = self.visual_branch(x)  # [b, p, d], [b, p, 1]

        textual_feat, textual_mask = self.textual_branch(x)  # [b, q, d], [b, q, 1]

        # TODO: mask can be computed separately from modules through a function that
        # verify a condition over "queries" or "proposals"
        # Please update in order to avoid passing masks as input

        scores, scores_mask = self.prediction_module(
            (visual_feat, visual_mask),
            (textual_feat, textual_mask),
            (concepts_pred, concepts_mask),
        )  # [b, q, b, p], [b, q, b, p]

        scores = scores.masked_fill(~scores_mask, 0)

        return scores, scores_mask

    def step(self, batch, batch_id):
        """
        :return: A tuple `(loss, metrics)`, where metrics is a dict with `acc`, `point_it`
        """
        queries = batch["queries"]  # [b, q, w]
        proposals = batch["proposals"]  # [b, p, 4]
        targets = batch["targets"]  # [b, q, 4]

        # TODO: refactor -> forward should directly return candidates ???
        scores, _ = self.forward(batch)  # [b, q, b, p]

        loss = self.loss({**batch, "scores": scores})

        candidates = self.predict_candidates(scores, proposals)  # [b, q, 4]

        acc = self.accuracy(
            candidates, targets, queries
        )  # TODO: refactor -> avoid passing queries whenever possible

        point_it = self.point_it(candidates, targets, queries)

        metrics = {
            "acc": acc,
            "point_it": point_it,
        }

        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self.step(batch, batch_idx)

        acc = metrics["acc"]
        point_it = metrics["point_it"]

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_pointit", point_it, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.step(batch, batch_idx)

        acc = metrics["acc"]
        point_it = metrics["point_it"]

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_pointit", point_it, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, metrics = self.step(batch, batch_idx)

        acc = metrics["acc"]
        point_it = metrics["point_it"]

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_pointit", point_it, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)

    def predict_candidates(self, scores, proposals):
        """
        Predict a candidate bounding box for each query

        :param scores: A tensor of shape [b, q, b, p] with the scores
        :param proposals: A tensor of shape [b, p, 4] with the proposals
        :return: A tensor of shape [b, q, 4] with the predicted candidates
        """
        b = scores.shape[0]
        q = scores.shape[1]
        p = scores.shape[3]

        index = torch.arange(b).to(self.device)  # [b]
        index = index.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [b, 1, 1, 1]
        index = index.repeat(1, q, 1, p)  # [b, q, 1, p]

        scores = scores.gather(-2, index).squeeze(-2)  # [b, q, p]

        scores = scores.argmax(-1).unsqueeze(-1).repeat(1, 1, 4)  # [b, q, 4]

        candidates = proposals.gather(-2, scores)  # [b, q, 4]

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

    def point_it(self, candidates, targets, queries):
        # candidates: [b, q, 4]
        # targets: [b, q, 4]
        # queries: [b, q, w]

        centers = tlbr2ctwh(candidates)[..., :2]  # [b, q, 2]

        topleft = targets[..., :2]  # [b, q, 2]
        bottomright = targets[..., 2:]  # [b, q, 2]
        
        # count a match whether center is inside the target
        matches = (centers >= topleft) & (centers <= bottomright)  # [b, q, 2]
        matches = matches.all(-1)  # [b, q]

        # mask padding queries
        _, is_query = get_queries_mask(queries)  # [b, q, w], [b, q]
        _, n_queries = get_queries_count(queries)  # [b, q], [b]

        matches = matches.masked_fill(~is_query, False)  # [b, q]

        point_it = matches.sum() / n_queries.sum()

        return point_it
        

class SimilarityPredictionModule(nn.Module):
    def __init__(self, omega):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.omega = omega
        # self.

    def forward(self, visual, textual, concepts):
        """
        :param visual: A tuple `(visual_feat, visual_mask)`, where `visual_feat` is a
            tensor of shape `[b, p, d]` and `visual_mask` is a tensor of shape `[b, p, 1]`

        :param textual: A tuple `(textual_feat, textual_mask)`, where `textual_feat`
            is a tensor of shape `[b, q, d]` and `textual_mask` is a tensor of
            shape `[b, q, 1]`

        :param concepts: A tuple `(concepts_pred, concepts_mask)`, where `concepts_pred`
            is a tensor of shape `[b, q, b, p]` and `concepts_mask` is a tensor of
            shape `[b, q, b, p]`
        """
        multimodal_pred, multimodal_mask = self.predict_multimodal(
            visual, textual
        )  # [b, q, b, p], [b, q, b, p]
        concepts_pred, concepts_mask = self.predict_concepts(
            concepts
        )  # [b, q, b, p], [b, q, b, p]

        scores = self.apply_prior(multimodal_pred, concepts_pred)  # [b, q, b, p]
        mask = multimodal_mask & concepts_mask  # [b, q, b, p]

        return scores, mask

    def predict_multimodal(self, visual, textual):
        visual_feat, visual_mask = visual  # [b, q, d], [b, q, 1]
        textual_feat, textual_mask = textual  # [b, p, d], [b, p, 1]

        b = textual_feat.shape[0]
        q = textual_feat.shape[1]
        p = visual_feat.shape[1]

        visual_feat, visual_mask = ext_visual(visual_feat, visual_mask, b, q, p)
        textual_feat, textual_mask = ext_textual(textual_feat, textual_mask, b, q, p)

        multimodal_mask = visual_mask & textual_mask  # [b, q, b, p]

        multimodal_pred = torch.cosine_similarity(
            visual_feat, textual_feat, dim=-1
        )  # [b, q, b, p]
        multimodal_pred = mask_softmax(multimodal_pred, multimodal_mask)  # [b, p, b, p]
        multimodal_pred = self.softmax(multimodal_pred)  # [b, p, b, p]

        return multimodal_pred, multimodal_mask

    def predict_concepts(self, concepts):
        concepts_pred, concepts_mask = concepts  # [b, q, b, p], [b, q, b, p]

        concepts_pred = mask_softmax(concepts_pred, concepts_mask)  # [b, q, b, p]
        concepts_pred = self.softmax(concepts_pred)  # [b, q, b, p]

        return concepts_pred, concepts_mask

    def apply_prior(self, predictions, prior):
        w = self.omega

        return w * predictions + (1 - w) * prior  # [b, q, b, p]


class NeuralNetworkPredictionModule(nn.Module):
    def __init__(self, omega):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(300 * 2, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, visual, textual, concepts):
        visual_feat, visual_mask = visual  # [b, q, d], [b, q, 1]
        textual_feat, textual_mask = textual  # [b, p, d], [b, p, 1]
        concepts_pred, concepts_mask = concepts  # [b, q, b, p], [b, q, b, p]

        b = textual_feat.shape[0]
        q = textual_feat.shape[1]
        p = visual_feat.shape[1]

        visual_feat, visual_mask = ext_visual(visual_feat, visual_mask, b, q, p)
        textual_feat, textual_mask = ext_textual(textual_feat, textual_mask, b, q, p)

        multimodal_mask = visual_mask & textual_mask  # [b, q, b, p]

        multimodal_feat = torch.cat(
            (visual_feat, textual_feat), dim=-1
        )  # [b, q, b, p, f], f = 2d

        multimodal_pred = self.linear(multimodal_feat).squeeze(-1)  # [b, q, b, p]
        multimodal_pred = mask_softmax(multimodal_pred, multimodal_mask)  # [b, q, b, p]
        multimodal_pred = self.softmax(multimodal_pred)  # [b, q, b, p]

        concepts_pred = mask_softmax(concepts_pred, concepts_mask)  # [b, q, b, p]
        concepts_pred = self.softmax(concepts_pred)  # [b, q, b, p]

        mask = multimodal_mask & concepts_mask  # [b, q, b, p]
        scores = self.apply_prior(multimodal_pred, concepts_pred)  # [b, q, b, p]

        scores = scores.masked_fill(~mask, 0)  # [b, q, b, p]

        return scores, mask

    def apply_prior(self, predictions, prior):
        w = self.omega

        return w * predictions + (1 - w) * prior  # [b, q, b, p]


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

        x1 = proposals[..., 0] / image_w  # [b, p]
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

        queries_x = queries_x.masked_fill(~mask, 0)

        return queries_x, mask
