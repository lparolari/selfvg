import unittest

import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity

from weakvg.model import TextualBranch, WordEmbedding
from weakvg.wordvec import get_wordvec


class TestWordEmbedding(unittest.TestCase):
    def test_word_embedding_similarity(self):
        wordvec, vocab = get_wordvec()
        we = WordEmbedding(wordvec, vocab)

        index = torch.tensor([vocab["man"], vocab["woman"], vocab["person"]])

        emb = we(index)

        man = emb[0]
        woman = emb[1]
        person = emb[2]

        self.assertAlmostEqual(
            cosine_similarity(man, person, dim=0).item(), 0.6443, places=4
        )
        self.assertAlmostEqual(
            cosine_similarity(woman, person, dim=0).item(), 0.6171, places=4
        )
        self.assertAlmostEqual(
            cosine_similarity(man, woman, dim=0).item(), 0.6999, places=4
        )


class TestTextualBranch(unittest.TestCase):
    wordvec, vocab = get_wordvec()
    we = WordEmbedding(wordvec, vocab)

    def test_lstm(self):
        queries_idx = [
            self.vocab.lookup_indices("the quick brown fox".split()),
            self.vocab.lookup_indices("the lazy dog <unk>".split()),
        ]
        queries = torch.tensor([queries_idx])

        textual_branch = TextualBranch(word_embedding=self.we)

        output, mask = textual_branch.forward({"queries": queries})

        self.assertEqual(output.shape, (1, 2, 300))
        self.assertEqual(mask.shape, (1, 2, 1))
