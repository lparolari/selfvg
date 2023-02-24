import unittest

from weakvg.padding import pad_labels_syn


class TestPadding(unittest.TestCase):
    def test_pad_labels_syn(self):
        labels_syn = [
            # first batch
            [
                # 3 proposals with up to 3 alternatives
                [1, 2, 3],
                [4, 5],
                [7],
            ],
            # second batch
            [
                # 2 proposals with up to 4 alternatives
                [11, 12, 13, 14],
                [15, 16],
            ],
        ]

        padded_labels_syn = pad_labels_syn(labels_syn, 4, 3)

        self.assertEqual(padded_labels_syn.shape, (2, 3, 4))  # [b, p, a]

        self.assertListEqual(padded_labels_syn[0, 0].tolist(), [1, 2, 3, 0])
        self.assertListEqual(padded_labels_syn[0, 1].tolist(), [4, 5, 0, 0])
        self.assertListEqual(padded_labels_syn[0, 2].tolist(), [7, 0, 0, 0])

        self.assertListEqual(
            padded_labels_syn[1, 0].tolist(), [11, 12, 13, 0]
        )  # last zero is because we set 3 to max_alternatives_length
        self.assertListEqual(padded_labels_syn[1, 1].tolist(), [15, 16, 0, 0])
        self.assertListEqual(padded_labels_syn[1, 2].tolist(), [0, 0, 0, 0])
