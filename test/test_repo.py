import unittest

from weakvg.repo import LabelsRepository

class TestLabelsRepository(unittest.TestCase):
    def test_get_synonyms(self):
        repo = LabelsRepository(
            labels=["man", "dog", "male", "dogs", "puppy"],
            alternatives=["1:man,3:male", "2:dog,4:dogs,5:puppy"],
        )

        alts4dog = repo.get_alternatives("dog")
        alts4puppy = repo.get_alternatives("puppy")

        self.assertListEqual(alts4dog, ["dog", "dogs", "puppy"])
        self.assertListEqual(alts4dog, alts4puppy)

    def test_from_vocab(self):
        labels_path = "data/objects_vocab.txt"
        alternatives_path = "data/objects_vocab_merged.txt"

        repo = LabelsRepository.from_vocab(labels_path, alternatives_path)

        self.assertEqual(len(repo.labels), 1600)
        self.assertEqual(len(repo.alternatives), 878)

        alts4dog = repo.get_alternatives("dog")
        alts4puppy = repo.get_alternatives("puppy")

        self.assertListEqual(alts4dog, ["dog", "dogs", "puppy"])
        self.assertListEqual(alts4dog, alts4puppy)
