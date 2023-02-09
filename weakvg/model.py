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
        proposals = batch['proposals']  # [b, p, 4]
        targets = batch['targets']  # [b, q, 4]
   
        scores, mask = self.forward(batch)  # [b, q, b, p]

        loss = self.loss(queries, scores)
        loss.requires_grad_()  # TODO: remove this (current workaround for https://discuss.pytorch.org/t/why-do-i-get-loss-does-not-require-grad-and-does-not-have-a-grad-fn/53145)

        acc = self.accuracy(scores, proposals, targets, queries)  # TODO: refactor -> avoid passing queries whenever possible
    
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def accuracy(self, scores, proposals, targets, queries):

        # TODO: this may be called `predict_step` (pl hook)
        b = scores.shape[0]
        q = scores.shape[1]
        p = scores.shape[3]

        index = torch.arange(b)  # [b]
        index = index.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [b, 1, 1, 1]
        index = index.repeat(1, q, 1, p)  # [b, q, 1, p]

        scores = scores.gather(-2, index).squeeze(-2)  # [b, q, p]

        pred = scores.argmax(-1).unsqueeze(-1).repeat(1, 1, 4)  # [b, q, 4]

        proposals = proposals.gather(-2, pred)  # [b, q, 4]

        targets  # [b, q, 4]

        # TODO: the following is a metric (unrelated to te computation of predictions)
        iou_threshold = 0.5

        from torchvision.ops import box_iou
        # iou_score = torchvision.ops.box_iou(proposals.view(-1, 4), targets.view(-1, 4)).view(b,q,b,q)

        # i1 = torch.arange(q).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(b, 1, b, q) # [b, 1, b, q]

        # iou_score = iou_score.gather(1, i1).squeeze(1) # [b, b, q]

        # i2 = torch.arange(b)

        iou_score = box_iou(proposals.view(-1, 4), targets.view(-1, 4))#  [b * q, b * q]

        index = torch.arange(b * q).unsqueeze(-1)  # [b * q, 1]

        iou_score = iou_score.gather(-1, index)  # [b * q, 1]

        iou_score = iou_score.view(b, q)  # [b, q]

        matches = iou_score >= iou_threshold  # [b, q]

        # mask as zero padding queries
        n_words = (queries != 0).sum(-1)  # [b, q]
        is_query = n_words > 0  # [b, q]
        n_queries = is_query.sum(-1).unsqueeze(-1)  # [b, 1]
       

        matches = matches.masked_fill(~is_query, False)

        acc = matches.sum() / n_queries.sum()

        return acc

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
