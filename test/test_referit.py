import unittest

from weakvg import repo
from weakvg.referit import ReferitDataset
from weakvg.wordvec import get_tokenizer, get_wordvec


class TestReferitDataset(unittest.TestCase):
    def setUp(self):
        super().setUp()

        _, self.vocab = get_wordvec()

        self.dataset = ReferitDataset(
            split="val",
            data_dir="data/referit",
            identifiers=[
                (10890, "10890_4"),
                (27604, "27604_2"),
                (3593, "3593_6"),
                (3593, "3593_5"),
            ],  # load few examples for testing purposes
            tokenizer=get_tokenizer(),
            vocab=self.vocab,
        )

    def test_load_samples_val(self):
        self.assertEqual(len(self.dataset), 4)

    def test_item_keys(self):
        required_keys = sorted(
            [
                "meta",
                "sentence",
                "queries",
                "heads",
                "image_w",
                "image_h",
                "proposals",
                "labels",
                "attrs",
                "labels_raw",
                "labels_syn",
                "proposals_feat",
                "targets",
            ]
        )

        keys = sorted(list(self.dataset.__getitem__(0).keys()))

        self.assertListEqual(keys, required_keys)

    def test_item_values(self):
        x = self.dataset.__getitem__(0)

        self.assertListEqual(x["meta"], [0, 10890])
        self.assertEqual(self._txt(x["sentence"]), "person on the left")
        self.assertEqual(self._txt(x["queries"][0]), "person on the left")
        self.assertEqual(self._txt(x["heads"][0]), "person")
        self.assertListEqual(x["targets"][0], [156.0, 169.0, 68.0, 191.0])
        self.assertEqual(x["image_w"], 480)
        self.assertEqual(x["image_h"], 360)
        self.assertEqual(len(x["proposals"]), 30)
        self.assertEqual(len(x["labels"]), 30)
        self.assertEqual(len(x["attrs"]), 178)
        self.assertEqual(len(x["labels_raw"]), 30)
        self.assertEqual(len(x["labels_syn"]), 30)
        self.assertEqual(len(x["proposals_feat"]), 30)

    def test_item_multiple_queries(self):
        x = self.dataset.__getitem__(1)

        self.assertListEqual(x["meta"], [1, 27604])

        self.assertEqual(len(x["queries"]), 2)

        self.assertEqual(self._txt(x["sentence"]), "top edge of roof")

        self.assertEqual(self._txt(x["queries"][0]), "top edge of roof")
        self.assertEqual(self._txt(x["heads"][0]), "edge roof")

        self.assertEqual(self._txt(x["queries"][1]), "rafters , directly below roof")
        self.assertEqual(self._txt(x["heads"][1]), "rafters roof")

    def test_can_open_repos(self):
        ds = self.dataset

        self.assertIsInstance(ds._open_refs_repo(), repo.RefsRepository)
        self.assertIsInstance(ds._open_annotations_repo(), repo.AnnotationsRepository)
        self.assertIsInstance(ds._open_images_size_repo(), repo.ImagesSizeRepository)
        self.assertIsInstance(
            ds._open_objects_detection_repo(), repo.ObjectsDetectionRepository
        )
        self.assertIsInstance(
            ds._open_objects_feature_repo(), repo.ObjectsFeature2Repository
        )
        self.assertIsInstance(ds._open_heads_repo(), repo.HeadsRepository)

    def _txt(self, indices):
        return " ".join(self.vocab.lookup_tokens(indices))
