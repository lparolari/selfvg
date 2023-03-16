import unittest

from weakvg.dataset.flickr30k import Flickr30kDataset, Flickr30kDatum


def _make_dataset(split):
    from weakvg.wordvec import get_tokenizer, get_wordvec

    tokenizer = get_tokenizer
    _, vocab = get_wordvec()

    return Flickr30kDataset(
        split, data_dir="data/flickr30k", tokenizer=tokenizer, vocab=vocab
    )


class TestFlickr30kDataset(unittest.TestCase):
    def test_load_samples_val(self):
        dataset = _make_dataset("val")

        self.assertEqual(len(dataset), 4986)

    def test_load_samples_test(self):
        dataset = _make_dataset("test")

        self.assertEqual(len(dataset), 4969)

    @unittest.skip("Skip train dataset as it is too large")
    def test_load_samples_train(self):
        dataset = _make_dataset("train")

        self.assertEqual(len(dataset), 0)  # TODO: get this number


class TestFlickr30kDatum(unittest.TestCase):
    def setUp(self):
        super().setUp()

        dataset = _make_dataset("test")

        images_size = dataset._open_images_size()
        objects_detection = dataset._open_objects_detection()
        objects_feature = dataset._open_objects_feature()

        self.precomputed = {
            "images_size": images_size,
            "objects_detection": objects_detection,
            "objects_feature": objects_feature,
        }

    def test_get_sentences_ann(self):
        sample = self._make_sample()

        sentences_ann = sample.get_sentences_ann()

        self.assertEqual(len(sentences_ann), 5)
        self.assertEqual(
            sentences_ann[0],
            "[/EN#283585/people A young white boy] with [/EN#283589/bodyparts short hair] smiling into [/EN#283586/other a microphone] , standing near [/EN#283584/people a slightly balding white male] smiling into [/EN#283587/other a microphone] .",
        )

    def test_get_sentence(self):
        sample = self._make_sample()

        sentence, sentence_ann = sample.get_sentence(0, return_ann=True)

        self.assertEqual(
            sentence,
            "a young white boy with short hair smiling into a microphone , standing near a slightly balding white male smiling into a microphone .",
        )

        self.assertEqual(
            sentence_ann,
            "[/EN#283585/people A young white boy] with [/EN#283589/bodyparts short hair] smiling into [/EN#283586/other a microphone] , standing near [/EN#283584/people a slightly balding white male] smiling into [/EN#283587/other a microphone] .",
        )

    def test_get_queries(self):
        sample = self._make_sample()

        queries, queries_ann = sample.get_queries(0, return_ann=True)

        self.assertEqual(len(queries), 5)
        self.assertEqual(len(queries_ann), 5)

        first_query = queries[0]
        first_query_ann = queries_ann[0]

        self.assertEqual(first_query, "a young white boy")
        self.assertEqual(first_query_ann, "/EN#283585/people A young white boy")

    def test_get_targets(self):
        sample = self._make_sample()

        # fmt: off
        target_283585 = [[8, 73, 286, 484, ]]
        target_283589 = [[8, 73, 164, 268, ], [263, 5, 435, 118, ]]
        target_283586 = [[191, 249, 337, 453, ], [353, 191, 447, 387, ]]
        target_283584 = [[177, 4, 500, 426, ]]
        target_283587 = [[353, 191, 447, 387, ], [197, 249, 327, 429,]]
        # fmt: on

        targets = [
            target_283585,
            target_283589,
            target_283586,
            target_283584,
            target_283587,
        ]

        self.assertListEqual(sample.get_targets(0), targets)

    def test_get_image_w(self):
        sample = self._make_sample()

        self.assertEqual(sample.get_image_w(), 500)

    def test_get_image_h(self):
        sample = self._make_sample()

        self.assertEqual(sample.get_image_h(), 486)

    def test_get_proposals(self):
        sample = self._make_sample()

        proposals = [
            [56, 0, 499, 485],
            [10, 65, 270, 330],
            [247, 10, 480, 276],
            [167, 187, 499, 485],
            [0, 287, 227, 485],
            [0, 47, 304, 485],
            [266, 6, 457, 150],
            [8, 69, 220, 232],
            [77, 207, 499, 485],
            [248, 1, 463, 259],
            [5, 63, 252, 318],
            [0, 30, 406, 485],
            [0, 0, 499, 269],
            [360, 178, 494, 481],
            [239, 180, 433, 485],
            [82, 166, 140, 233],
            [264, 113, 304, 184],
            [279, 33, 441, 243],
            [0, 41, 300, 485],
            [353, 238, 463, 357],
            [193, 256, 305, 378],
            [348, 134, 388, 183],
            [197, 158, 231, 200],
            [344, 179, 404, 210],
            [261, 117, 305, 184],
            [316, 125, 349, 147],
            [378, 112, 409, 135],
            [310, 117, 414, 155],
            [374, 106, 410, 128],
            [315, 117, 354, 138],
            [341, 175, 411, 207],
            [0, 294, 221, 485],
            [353, 194, 417, 263],
        ]

        self.assertListEqual(sample.get_proposals(), proposals)

    def test_get_labels(self):
        sample = self._make_sample()

        classes = [
            "man",
            "man",
            "man",
            "shirt",
            "shirt",
            "woman",
            "hair",
            "hair",
            "jacket",
            "head",
            "head",
            "boy",
            "wall",
            "fork",
            "tie",
            "ear",
            "ear",
            "face",
            "person",
            "hand",
            "hand",
            "nose",
            "nose",
            "mouth",
            "phone",
            "eye",
            "eye",
            "eyes",
            "eyebrow",
            "eyebrow",
            "mustache",
            "tshirt,t-shirt,t shirt",
            "microphone",
        ]

        self.assertListEqual(sample.get_labels(), classes)

    def test_get_attrs(self):
        sample = self._make_sample()

        attrs = [
            "white",
            "white",
            "white",
            "green",
            "blue",
            "smiling",
            "brown",
            "gray",
            "yellow",
            "white",
            "white",
            "white",
            "white",
            "black",
            "green",
            "pink",
            "pink",
            "short",
            "smiling",
            "black",
            "white",
            "long",
            "long",
            "red",
            "pink",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "red",
            "blue",
            "black",
        ]

        self.assertListEqual(sample.get_attrs(), attrs)

    def test_get_labels_syn(self):
        sample = self._make_sample()

        self.assertListEqual(
            sample.get_labels_syn(), [[label] for label in sample.get_labels()]
        )

    def test_get_proposals_feat(self):
        sample = self._make_sample()

        # identifier 92679312 --> imgid2idx --> idx 925
        # idx 925 --> pos_bboxes --> pos [37497, 37530]
        # pos [37497, 37530] --> features[pos[0]:pos[1]] --> feats (33, 2048)

        proposals_feat_shape = (33, 2048)

        self.assertEqual(sample.get_proposals_feat().shape, proposals_feat_shape)

    def _make_sample(self):
        return Flickr30kDatum(
            92679312, data_dir="data/flickr30k", precomputed=self.precomputed
        )
