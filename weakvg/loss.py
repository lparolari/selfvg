import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scores, targets):
        b = scores.shape[0]

        positive_index = targets.unsqueeze(-1)  # [b, 1]
        negative_index = (targets.unsqueeze(-1) + 1) % b  # [b, 1]

        scores_p = scores.gather(-1, positive_index).squeeze(-1)  # [b]
        scores_n = scores.gather(-1, negative_index).squeeze(-1)  # [b]

        return -scores_p.mean() + scores_n.mean()
