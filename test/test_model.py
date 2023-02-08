import unittest
import torch
import torch.nn as nn
import torch
from torchtext.vocab import GloVe, Vocab, vocab as make_vocab

from weakvg.wordvec import get_wordvec
from weakvg.model import WordEmbedding
from torch.nn.functional import cosine_similarity


class TestWordEmbedding(unittest.TestCase):
    def test_1(self):
        wordvec, vocab = get_wordvec()
        wv = WordEmbedding(wordvec, vocab)

        index = torch.tensor([vocab["man"], vocab["woman"], vocab["person"]])

        emb = wv(index)

        man = emb[0]
        woman = emb[1]
        person = emb[2]

        self.assertAlmostEqual(cosine_similarity(man, person, dim=0).item(), 0.6443, places=4)
        self.assertAlmostEqual(cosine_similarity(woman, person, dim=0).item(), 0.6171, places=4)
        self.assertAlmostEqual(cosine_similarity(man, woman, dim=0).item(), 0.6999, places=4)
