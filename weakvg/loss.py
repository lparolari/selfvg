import torch
import torch.nn as nn

from weakvg.utils import get_queries_mask, get_queries_count


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        queries = x["queries"]  # [b, q, w]
        scores = x["scores"]  # [b, q, b, p]

        b = scores.shape[0]

        n_words = (queries != 0).sum(-1)  # [b, q]
        is_query = n_words > 0  # [b, q]
        n_queries = is_query.sum(-1).unsqueeze(-1)  # [b, 1]
        has_query = is_query.any(-1).unsqueeze(-1)  # [b, 1]

        scores, _ = scores.max(-1)  # [b, q, b]
        scores = scores.sum(-2) / n_queries  # [b, b]
        scores = scores.masked_fill(~has_query, 0)

        targets = torch.eye(b).to(queries.device)  # [b, b]
        targets = targets.argmax(-1)  # [b]

        positive_index = targets.unsqueeze(-1)  # [b, 1]
        negative_index = (targets.unsqueeze(-1) + 1) % b  # [b, 1]

        scores_p = scores.gather(-1, positive_index).squeeze(-1)  # [b]
        scores_n = scores.gather(-1, negative_index).squeeze(-1)  # [b]

        return -scores_p.mean() + scores_n.mean()


class LossSupervised(nn.Module):
    def __init__(self):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, x):
        scores = x["scores"]  # [b, q, b, p]
        proposals = x["proposals"]  # [b, p, 4]
        targets = x["targets"]  # [b, q, 4]
        queries = x["queries"]  # [b, q, w]

        scores = self.positive(scores)  # [b, q, p]
        classes = self.classes(targets, proposals)  # [b, q]

        # cross entropy loss requires the input tensor to have shape
        # [N, C, d1, ..., dk] where C is the number of classes, while
        # the target tensor to have shape [N, d1, ..., dk]
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

        scores = scores.permute(0, 2, 1)  # [b, p, q]

        loss = self.ce(scores, classes)  # [b, q]

        _, is_query = get_queries_mask(queries)  # [b, q]
        _, n_queries = get_queries_count(queries)  # [b]

        loss = loss.masked_fill(~is_query, 0)  # [b, q]
        loss = loss.sum() / n_queries.sum()

        return loss

    def positive(self, scores):
        """
        Gather the scores of the positive proposals.

        :param scores: Tensor with shape `[b, q, b, p]`
        :return: Tensor with shape `[b, q, p]`
        """
        b = scores.shape[0]
        q = scores.shape[1]
        p = scores.shape[3]

        ident = torch.eye(b).to(scores.device)  # [b, b]
        ident = ident.argmax(-1)  # [b]
        ident = ident.view(-1, 1, 1, 1).repeat(1, q, 1, p)  # [b, q, 1, p]

        scores = scores.gather(-2, ident).squeeze(-2)  # [b, q, p]

        return scores

    def classes(self, targets, proposals):
        """
        Return the index of the proposal with maximum intersection
        over union wrt targets

        :param targets: A tensor `[b, q, 4]`
        :param proposals: A tensor `[b, p, 4]`
        :return: A tensor `[b, q]`
        """
        from torchvision.ops import box_iou

        b = targets.shape[0]
        q = targets.shape[1]
        p = proposals.shape[1]

        targets = targets.reshape(-1, 4)  # [b * q, 4]
        proposals = proposals.reshape(-1, 4)  # [b * p, 4]

        scores = box_iou(targets, proposals)  # [b * q, b * p]

        scores = scores.reshape(-1, b, p)  # [b * q, b, p]
        classes = scores.argmax(-1)  # [b * q, b]
        classes = classes.reshape(b, q, b)  # [b, q, b]

        index = torch.arange(b).to(targets.device)  # [b]
        index = index.unsqueeze(-1).unsqueeze(-1).repeat(1, q, 1)  # [b, q, 1]

        classes = classes.gather(-1, index).squeeze(-1)  # [b, q]

        return classes
