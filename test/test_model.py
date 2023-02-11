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


class TestConceptBranch(unittest.TestCase):
    wordvec, vocab = get_wordvec()
    we = WordEmbedding(wordvec, vocab)

    def test_forward(self):
        heads = torch.tensor(
            [
                self.vocab.lookup_indices("man <unk>".split()),
                self.vocab.lookup_indices("dog <unk>".split()),
            ]
        ).unsqueeze(
            0
        )  # [b, q, h] = [1, 2, 2]

        labels = torch.tensor(
            [
                self.vocab["person"],
                self.vocab["dog"],
                self.vocab["sky"],
                self.vocab["person"],
                self.vocab["bird"],
                self.vocab.get_default_index(),  # pad
                self.vocab.get_default_index(),  # pad
                self.vocab.get_default_index(),  # pad
            ]
        ).unsqueeze(
            0
        )  # [b, p] = [1, 8]

        concept_branch = ConceptBranch(word_embedding=self.we)

        output, mask = concept_branch.forward({"heads": heads, "labels": labels})

        self.assertEqual(mask.shape, (1, 2, 1, 8))
        self.assertEqual(output.shape, (1, 2, 1, 8))

        q1 = (0, 0, 0)
        q2 = (0, 1, 0)

        self.assertAlmostEqual(output[q1].argmax().item(), 0)  # also label 4 could be correct, but the argmax selects the first one
        self.assertAlmostEqual(output[q2].argmax().item(), 1)

        self.assertAlmostEqual(mask[q1].sum(), 5)
        self.assertAlmostEqual(mask[q2].sum(), 5)
