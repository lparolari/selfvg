import unittest

import torch
from torch.nn.functional import cosine_similarity

from weakvg.model import TextualBranch, WordEmbedding, VisualBranch, ConceptBranch
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


class TestVisualBranch(unittest.TestCase):
    wordvec, vocab = get_wordvec()
    we = WordEmbedding(wordvec, vocab)

    def test_spatial(self):
        proposals = torch.tensor([[[20, 20, 80, 100]]])  # [b, p, 4]
        image_w = torch.tensor([500])
        image_h = torch.tensor([335])

        visual_branch = VisualBranch(word_embedding=self.we)

        spat = visual_branch.spatial(
            {
                "proposals": proposals,
                "image_w": image_w,
                "image_h": image_h,
            }
        )

        self.assertEqual(spat.shape, (1, 1, 5))

        self.assertAlmostEqual(spat[0, 0, 0].item(), 20 / 500)
        self.assertAlmostEqual(spat[0, 0, 1].item(), 20 / 335)
        self.assertAlmostEqual(spat[0, 0, 2].item(), 80 / 500)
        self.assertAlmostEqual(spat[0, 0, 3].item(), 100 / 335)
        self.assertAlmostEqual(
            spat[0, 0, 4].item(), ((80 - 20) * (100 - 20)) / (500 * 335)
        )

    def test_project(self):
        proposals_feat = torch.rand(2, 6, 3)  # [b, p, v]
        spat = torch.rand(2, 6, 5).mul(100).long()  # [b, p, 5]

        visual_branch = VisualBranch(word_embedding=self.we)

        proj = visual_branch.project(proposals_feat, spat)

        self.assertEqual(proj.shape, (2, 6, 8))
