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

    index = torch.arange(b * q).to(candidates.device).unsqueeze(-1)  # [b * q, 1]

    scores = scores.gather(-1, index)  # [b * q, 1]
    scores = scores.view(b, q)  # [b, q]

    return scores


def get_queries_mask(queries):
    """
    Return a mask for the words and queries in a batch of queries.

    :param queries: A tensor with shape `[b, q, w]`
    :return: A tuple of tensors `([b, q, w], [b, q])` for is_word, is_query
    """
    is_word = queries != 0  # [b, q, w]
    is_query = is_word.any(-1)  # [b, q]

    return is_word, is_query


def get_queries_count(queries):
    """
    Return the number of words and queries in a batch of queries.

    :param queries: A tensor with shape `[b, q, w]`
    :return: A tuple of tensors `([b, q], [b])` for n_words, n_queries
    """
    is_word, is_query = get_queries_mask(queries)

    n_words = is_word.sum(-1)  # [b, q]
    n_queries = is_query.sum(-1)  # [b]

    return n_words, n_queries


def get_proposals_mask(proposals):
    """
    Return a mask for the proposals in a batch of proposals.

    :proposals: A tensor with shape `[b, p, 4]`
    :return: A tensor with shape `[b, p]`
    """
    return proposals.greater(0).any(-1)


def get_multimodal_mask(queries, proposals):
    """
    :param queries: A tensor with shape `[b, q, w]`
    :param proposals: A tensor with shape `[b, p, 4]`
    :return: A tensor with shape `[b, q, b, p]`
    """
    queries_mask = get_queries_mask(queries)[1]  # [b, q]
    proposals_mask = get_proposals_mask(proposals)  # [b, p]

    b = queries.shape[0]
    q = queries.shape[1]
    p = proposals.shape[1]

    queries_mask = queries_mask.view(b, q, 1, 1).repeat(1, 1, b, p)  # [b, q, b, p]
    proposals_mask = proposals_mask.view(1, 1, b, p).repeat(b, q, 1, 1)  # [b, q, b, p]

    return queries_mask & proposals_mask


def get_concepts_mask(heads, labels):
    """
    :param heads: A tensor with shape `[b, q, h]`
    :param labels: A tensor with shape `[b, p]`
    :return: A tensor with shape `[b, q, b, p]`
    """
    n_heads = (heads != 0).sum(-1).unsqueeze(-1)  # [b, q, 1]

    has_head = (n_heads != 0).unsqueeze(-1)  # [b, q, 1, 1]
    has_label = (labels != 0).unsqueeze(0).unsqueeze(0)  # [1, 1, b, p]

    mask = has_head & has_label

    return mask


def get_mask(x):
    """
    :param x: A dict with keys `queries`, `proposals`, `heads`, `labels`
    :return: A tensor with shape `[b, q, b, p]`
    """
    queries = x["queries"]
    proposals = x["proposals"]
    heads = x["heads"]
    labels = x["labels"]

    multimodal_mask = get_multimodal_mask(queries, proposals)
    concepts_mask = get_concepts_mask(heads, labels)

    return multimodal_mask & concepts_mask


def ext_visual(visual_feat, visual_mask, b, q, p):
    """
    Extend given input to get shape `[b, q, b, p, d]` for visual_feat and `[b, q, b, p]` for visual_mask.

    :param visual_feat: A tensor  of shape `[b, p, d]`
    :param visual_mask: A tensor of shape `[b, p]`
    :return: A tuple of tensors `([b, q, b, p, d], [b, q, b, p])` for visual_feat, visual_mask
    """
    resh = 1, 1, b, p
    rep = b, q, 1, 1

    visual_feat = visual_feat.reshape(*resh, -1).repeat(*rep, 1)  # [b, q, b, p, d]
    visual_mask = visual_mask.reshape(*resh).repeat(*rep)  # [b, q, b, p]

    return visual_feat, visual_mask


def ext_textual(textual_feat, textual_mask, b, q, p):
    """
    Extend given input to get shape `[b, q, b, p, d]` for textual_feat and `[b, q, b, p]` for textual_mask.

    :param textual_feat: A tensor  of shape `[b, q, d]`
    :param textual_mask: A tensor of shape `[b, q]`
    :return: A tuple of tensors `([b, q, b, p, d], [b, q, b, p])` for textual_feat, textual_mask
    """
    resh = b, q, 1, 1
    rep = 1, 1, b, p

    textual_feat = textual_feat.reshape(*resh, -1).repeat(*rep, 1)  # [b, q, b, p, d]
    textual_mask = textual_mask.reshape(*resh).repeat(*rep)  # [b, q, b, p]

    return textual_feat, textual_mask


def mask_softmax(x, mask):
    # masking to -1e8 is required to enforce softmax predictions to be 0 for
    # masked values
    return x.masked_fill(~mask, -1e8)  # [b, p, b, p]


def tlbr2tlwh(x):
    """
    Convert bounding boxes from [x1, y1, x2, y2] to [x1, y1, w, h]

    :param x: A tensor of shape `[*, 4]`
    :return: A tensor of shape `[*, 4]`
    """
    x1, y1, x2, y2 = x.unbind(-1)
    w = x2 - x1
    h = y2 - y1
    return torch.stack([x1, y1, w, h], -1)


def tlbr2ctwh(x):
    """
    Convert bounding boxes from [x1, y1, x2, y2] to [cx, cy, w, h] where (cx, cy) is
    the center of the box

    :param x: A tensor of shape `[*, 4]`
    :return: A tensor of shape `[*, 4]`
    """
    x1, y1, w, h = tlbr2tlwh(x).unbind(-1)
    cx = x1 + w / 2
    cy = y1 + h / 2
    return torch.stack([cx, cy, w, h], -1)


def ctwh2tlbr(x):
    """
    Convert bounding boxes from [cx, cy, w, h] to [x1, y1, x2, y2]

    :param x: A tensor of shape `[*, 4]`
    :return: A tensor of shape `[*, 4]`
    """
    cx, cy, w, h = x.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], -1)
