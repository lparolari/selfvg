import torch
import torch.nn as nn

from weakvg.masking import get_queries_count


class Loss(nn.Module):
    def __init__(self, word_embedding, neg_selection="random"):
        """
        :param neg_selection: Negative selection strategy. One of:
          * `random`, selects one random negative example
          * `textual_sim_max`, selects the most similar example according
            to the averaged similarity of queries among examples.
        """
        super().__init__()
        self.we = word_embedding
        self.neg_selection = neg_selection

    def forward(self, x):
        queries = x["queries"]  # [b, q, w]
        scores = x["scores"]  # [b, q, b, p]

        queries_mask = (queries != 0) # [b, q, w]
        n_words = queries_mask.sum(-1)  # [b, q]
        is_query = n_words > 0  # [b, q]
        n_queries = is_query.sum(-1).unsqueeze(-1)  # [b, 1]
        has_query = is_query.any(-1).unsqueeze(-1)  # [b, 1]

        queries_e = self.we(queries, queries_mask)  # [b, q, w, d]

        x = {**x, "queries_e": queries_e}

        scores, _ = scores.max(-1)  # [b, q, b]
        scores = scores.sum(-2) / n_queries  # [b, b]
        scores = scores.masked_fill(~has_query, 0)

        pos = self.select_pos(x)  # [b, b]
        neg = self.select_neg(x, pos)  # [b, b]

        scores_p = (scores * pos).sum(-1) / pos.sum(-1)  # [b]
        scores_n = (scores * neg).sum(-1) / neg.sum(-1)  # [b]

        return -scores_p.mean() + scores_n.mean()

    def select_pos(self, x):
        scores = x["scores"]  # [b, q, b, p]

        b = scores.shape[0]

        return torch.eye(b).to(scores.device).bool()  # [b, b]

    def select_neg(self, x, pos):
        """
        :param x: Batch dict
        :param pos: A tensor of shape `[b, b]` with positive mask
        :return: A tensor of shape `[b, b]` with negative mask
        """
        queries = x["queries"]  # [b, q, w]
        queries_e = x["queries_e"]  # [b, q, w, d]

        if self.neg_selection == "textual_sim_max":
            # select the most similar example

            n_words, n_queries = get_queries_count(queries)  # [b, q], [b]

            sim = self.textual_sim(queries_e, n_words, n_queries)  # [b, b]
            sim = (sim + 1) / 2  # [b, b]

            # filter positive examples, which have similarity 1
            sim = sim.masked_fill(pos, 0)  # [b, b]

            # negative similarity, i.e., the similarity value of the most
            # similar example
            sim_n, _ = sim.max(-1, keepdim=True)  # [b, 1]

            mask = sim == sim_n  # [b, b]

            return mask

        # default to random selection
        return pos.roll(-1, dims=0)  # [b, b]

    def textual_sim(self, queries_e, n_words, n_queries):
        """
        Returns a tensor of shape `[b, b]` with the averaged similarity between examples.
        """
        b = queries_e.shape[0]

        # averaged query representation over words
        query_repr = queries_e.sum(dim=-2) / n_words.unsqueeze(-1)  # [b, q, d]
        query_repr = query_repr.masked_fill(
            (n_words == 0).unsqueeze(-1), value=0
        )  # [b, q, d]

        # averaged query representation over phrases
        query_repr = query_repr.sum(dim=-2) / n_queries.unsqueeze(-1)  # [b, d]
        query_repr = query_repr.masked_fill(
            (n_queries == 0).unsqueeze(-1), value=0
        )  # [b, d]

        query_repr_a = query_repr.unsqueeze(1).repeat(1, b, 1)  # [b, b, d]
        query_repr_b = query_repr.unsqueeze(0).repeat(b, 1, 1)  # [b, b, d]

        query_similarity = torch.cosine_similarity(
            query_repr_a, query_repr_b, dim=-1
        )  # [b, b]

        return query_similarity
