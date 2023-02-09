import unittest

from weakvg.wordvec import fix_oov, get_wordvec, network_mask


class WordvecTestCase(unittest.TestCase):
    def test_fix_oov(self):
        wordvec, vocab = get_wordvec(custom_tokens=[])
        possibly_oov_labels = self._get_objects_vocab()

        missing = fix_oov(possibly_oov_labels, wordvec=wordvec, vocab=vocab)

        self.assertEqual(len(missing), 299)

        # check label "alarm clock" was fixed using embeddings from both
        # known words "alarm" and "clock"
        alarm_clock = missing[0]

        self.assertEqual(alarm_clock[0], "alarm clock")
        self.assertEqual(alarm_clock[1], ["alarm", "clock"])

        # check label "microwave,microwave oven" was fixed using microwave
        # known word
        microwave = missing[5]

        self.assertEqual(microwave[0], "microwave,microwave oven")
        self.assertEqual(microwave[1], ["microwave"])

    def test_get_wordvec(self):
        wordvec, vocab = get_wordvec(custom_tokens=[])

        self.assertEqual(len(vocab), 400000 + 1)  # +1 for padding
        self.assertEqual(len(wordvec), 400000 + 1)

    def test_get_wordvec__custom_tokens(self):
        wordvec, vocab = get_wordvec(custom_tokens=network_mask())

        self.assertEqual(
            len(vocab), 400000 + 299 + 1
        )  # +299 for fixed oov, +1 for padding
        self.assertEqual(len(wordvec), 400000 + 299 + 1)
