from typing import List

import torch
from torchvision.ops import box_iou

Box = List[int]


def union_box(boxes: List[Box]):
    x1 = min([box[0] for box in boxes])
    y1 = min([box[1] for box in boxes])
    x2 = max([box[2] for box in boxes])
    y2 = max([box[3] for box in boxes])
    return [x1, y1, x2, y2]


def iou(candidates, targets):
    # TODO: re-implement the following code using https://torchmetrics.readthedocs.io/en/stable/detection/mean_average_precision.html

    # proposals: [b, q, 4]
    # targets: [b, q, 4]

    b = candidates.shape[0]
    q = candidates.shape[1]

    scores = box_iou(candidates.view(-1, 4), targets.view(-1, 4))  #  [b * q, b * q]

    index = torch.arange(b * q).unsqueeze(-1)  # [b * q, 1]

    scores = scores.gather(-1, index)  # [b * q, 1]
    scores = scores.view(b, q)  # [b, q]

    return scores
