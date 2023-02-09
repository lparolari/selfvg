import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, queries, scores):
        b = scores.shape[0]

        n_words = (queries != 0).sum(-1)  # [b, q]
        is_query = n_words > 0  # [b, q]
        n_queries = is_query.sum(-1).unsqueeze(-1)  # [b, 1]

        scores, _ = scores.max(-1)  # [b, q, b]
        scores = scores.sum(-2) / n_queries  # [b, b]

        targets = torch.eye(b)  # [b, b]
        targets = targets.argmax(-1)  # [b]

        positive_index = targets.unsqueeze(-1)  # [b, 1]
        negative_index = (targets.unsqueeze(-1) + 1) % b  # [b, 1]

        scores_p = scores.gather(-1, positive_index).squeeze(-1)  # [b]
        scores_n = scores.gather(-1, negative_index).squeeze(-1)  # [b]

        return -scores_p.mean() + scores_n.mean()
