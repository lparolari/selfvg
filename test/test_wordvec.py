import unittest

from weakvg.wordvec import get_wordvec, fix_oov

class WordvecTestCase(unittest.TestCase):
    def test_fix_oov(self):
        wordvec, vocab = get_wordvec(check_oov=[])
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
        wordvec, vocab = get_wordvec(check_oov=[])
  
        self.assertEqual(len(vocab), 400000)
        self.assertEqual(len(wordvec), 400000)
    
    def test_get_wordvec__check_oov(self):
        wordvec, vocab = get_wordvec(check_oov=self._get_objects_vocab())
  
        self.assertEqual(len(vocab), 400299)
        self.assertEqual(len(wordvec), 400299)

    def _get_objects_vocab(self):
        with open("data/objects_vocab.txt", "r") as f:
            labels = f.readlines()
            labels = [label.strip() for label in labels]
        return labels