import unittest

import torch

from weakvg.wordvec import WordvecBuilder, get_wordvec, get_objects_vocab


class TestWordvec(unittest.TestCase):
    def test_get_wordvec(self):
        wordvec, vocab = get_wordvec(dim=50, custom_labels=[])

        self.assertEqual(len(vocab), 400000 + 1)  # +1 for padding
        self.assertEqual(len(wordvec), 400000 + 1)

        self.assertEqual(wordvec.vectors.shape, (400000 + 1, 50))

    def test_get_objects_vocab(self):
        self.assertEqual(len(get_objects_vocab()), 1600)


class TestWordvecBuilder(unittest.TestCase):
    dim = 50
    wv_kwargs = {"name": "6B", "dim": dim, "cache": "data/glove"}

    def test_with_glove(self):
        d = self.dim
        wv = WordvecBuilder().with_glove(**self.wv_kwargs).get_wordvec()
        vecs = wv.vectors

        self.assertEqual(vecs.shape, (400000, d))

    def test_with_vocab(self):
        d = self.dim
        b = WordvecBuilder().with_glove(**self.wv_kwargs).with_vocab()

        wv = b.get_wordvec()
        vecs = wv.vectors
        vocab = b.get_vocab()

        self.assertEqual(vecs.shape, (400001, d))

        self.assertEqual(len(wv), 400001)
        self.assertEqual(len(vocab), 400001)  # +1 for <unk>

        self.assertTrue(torch.equal(wv["<unk>"], torch.zeros(d)))
        self.assertTrue(torch.equal(wv["<unk>"], vecs[vocab["<unk>"]]))
        self.assertTrue(torch.equal(wv["the"], vecs[vocab["the"]]))

        self.assertEqual(wv.stoi["<unk>"], 0)
        self.assertEqual(wv.stoi["the"], 1)

        self.assertEqual(vocab.get_default_index(), 0)

        self.assertEqual(vocab["<unk>"], 0)

        self.assertEqual(vocab.lookup_token(0), "<unk>")
        self.assertEqual(vocab.lookup_token(1), "the")

    def test_with_vocab__raises_whether_wordvec_not_build(self):
        with self.assertRaises(AssertionError):
            WordvecBuilder().with_vocab()

    def test_with_custom_labels__known(self):
        b = (
            WordvecBuilder()
            .with_glove(**self.wv_kwargs)
            .with_vocab()
            .with_custom_labels(["microwave"])
        )

        wv = b.get_wordvec()
        vocab = b.get_vocab()

        self.assertEqual(len(wv), 400001)
        self.assertEqual(len(vocab), 400001)

    def test_with_custom_labels__unknown(self):
        sim = torch.cosine_similarity

        b = (
            WordvecBuilder()
            .with_glove(**self.wv_kwargs)
            .with_vocab()
            .with_custom_labels(["microwave oven"])
        )

        wv = b.get_wordvec()
        vecs = wv.vectors
        vocab = b.get_vocab()

        self.assertEqual(len(wv), 400002)  # +1 unk, +1 added
        self.assertEqual(len(vocab), 400002)

        self.assertEqual(vocab["microwave oven"], 400001)  # vocab should be updated accordingly

        self.assertTrue(torch.equal(wv["microwave oven"], vecs[-1]))  # should be last added
        
        # "microwave oven" is build by averaging "microwave" and "oven", thus
        # we perform a sanity check through cosine similarity
        self.assertAlmostEqual(sim(wv["microwave oven"], wv["microwave"], dim=0).item(), 0.9121, places=4)
        self.assertAlmostEqual(sim(wv["microwave oven"], wv["oven"], dim=0).item(), 0.9370, places=4)

    def test_with_custom_labels__from_vocab(self):
        d = self.dim
        b = (
            WordvecBuilder()
            .with_glove(**self.wv_kwargs)
            .with_vocab()
            .with_custom_labels(get_objects_vocab())
        )

        wv = b.get_wordvec()
        vecs = wv.vectors
        vocab = b.get_vocab()

        self.assertEqual(vecs.shape, (400000 + 1 + 299, d))  # +1 oov, +299 added

        self.assertTrue(torch.equal(wv["the"], vecs[vocab["the"]]))
        self.assertTrue(torch.equal(wv["microwave,microwave oven"], vecs[vocab["microwave,microwave oven"]]))

    def test_with_custom_labels__not_breaking_prev_structure(self):
        d = self.dim
        b = (
            WordvecBuilder()
            .with_glove(**self.wv_kwargs)
            .with_vocab()
            .with_custom_labels(["microwave oven"])
        )

        wv = b.get_wordvec()
        vecs = wv.vectors

        self.assertTrue(torch.equal(vecs[0], torch.zeros(d)))

    def test_with_custom_tokens__known(self):
        d = self.dim
        b = (
            WordvecBuilder()
            .with_glove(**self.wv_kwargs)
            .with_vocab()
            .with_custom_tokens(["dog"])
        )

        wv = b.get_wordvec()
        vocab = b.get_vocab()

        self.assertEqual(len(wv), 400000 + 1)  # +1 oov
        self.assertEqual(len(vocab), 400000 + 1)
    
    def test_with_custom_tokens__unknown(self):
        d = self.dim
        b = (
            WordvecBuilder()
            .with_glove(**self.wv_kwargs)
            .with_vocab()
            .with_custom_tokens(["blabla"])
        )

        wv = b.get_wordvec()
        vocab = b.get_vocab()

        self.assertEqual(len(wv), 400000 + 1 + 1)  # +1 oov, +1 added
        self.assertEqual(len(vocab), 400000 + 1 + 1)
        self.assertTrue("blabla" in vocab)
        self.assertTrue(~torch.equal(wv["blabla"], torch.zeros(d)))

    def test_with_custom_tokens__not_breaking_prev_structure(self):
        d = self.dim
        b = (
            WordvecBuilder()
            .with_glove(**self.wv_kwargs)
            .with_vocab()
            .with_custom_labels(["microwave oven"])
        )

        wv = b.get_wordvec()
        vecs = wv.vectors

        self.assertTrue(torch.equal(vecs[0], torch.zeros(d)))
