import unittest

from weakvg.datamodule import WeakvgDataModule
from weakvg.wordvec import get_nlp, get_tokenizer, get_wordvec


class TestWeakvgDataModule(unittest.TestCase):
    def test_test_dataloader(self):
        _, vocab = get_wordvec()

        dm = WeakvgDataModule(
            data_dir="data/flickr30k",
            batch_size=2,
            num_workers=1,
            tokenizer=get_tokenizer(),
            vocab=vocab,
            nlp=get_nlp(),
        )

        dm.setup("test")  # should load test dataset

        test_dl = dm.test_dataloader()

        self.assertEqual(
            len(test_dl), 2484
        )  # (5000 samples - some discarded samples)  / 2 batch size

        # assert some properties of the batch

        batch = next(iter(test_dl))

        queries = batch["queries"]
        targets = batch["targets"]

        self.assertEqual(queries.shape[1], targets.shape[1])  # n queries == n targets

        self.assertEqual(queries.shape[-1], 12)  # n words in query
        self.assertEqual(
            targets.shape[-1], 4
        )  # target is a bounding box, i.e. tensor of 4 values x1, y1, x2, y2

        proposals = batch["proposals"]
        proposals_feat = batch["proposals_feat"]
        labels = batch["labels"]
        attrs = batch["attrs"]

        self.assertEqual(proposals.shape[1], proposals_feat.shape[1])  # n proposals
        self.assertEqual(proposals.shape[1], labels.shape[1])  # n proposals
        self.assertEqual(proposals.shape[1], attrs.shape[1])  # n proposals

        self.assertEqual(proposals.shape[-1], 4)  # bounding box x1, y1, x2, y2
        self.assertEqual(proposals_feat.shape[-1], 2048)  # features
