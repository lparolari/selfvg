import logging
import os
from typing import List

from torch.utils.data import Dataset

from weakvg.dataset.mixin import UpperboundAccuracyMixin, QueryNumberMixin
from weakvg.repo import (
    AnnotationsRepository,
    HeadsRepository,
    ImagesSizeRepository,
    LabelsRepository,
    ObjectsDetectionRepository,
    ObjectsFeature2Repository,
    RefsRepository,
)
from weakvg.dataset.shared import get_locations, get_relations


Feature = List[float]
Box = List[float]
Attr = str
Label = str
Head = str
Query = [str]
Sentence = str
AnnId = str
ImageId = int


class ReferitSample:
    def __init__(self, **kwargs):
        self.image_id: ImageId = kwargs["image_id"]
        """An int representing the image identifier"""

        self.ann_id: AnnId = kwargs["ann_id"]
        """An int representing the annotation identifier"""

        self.sentence: Sentence = kwargs["sentence"]
        """A string representing the sentence"""

        self.queries: List[Query] = kwargs["queries"]
        """A list of strings representing the queries"""

        self.heads: List[Head] = kwargs["heads"]
        """A list strings representing query's heads. Every head could be a single
           word or multiple words separated by a space."""

        self.image_w: int = kwargs["image_w"]
        """An int representing the image width"""

        self.image_h: int = kwargs["image_h"]
        """An int representing the image height"""

        self.proposals: List[Box] = kwargs["proposals"]
        """A list of lists of ints representing proposals' bounding boxes"""

        self.labels: List[str] = kwargs["labels"]
        """A list of strings representing the labels, one for each proposal"""

        self.attrs: List[Attr] = kwargs["attrs"]
        """A list of strings representing the attributes, one for each proposal"""

        self.labels_raw = kwargs["labels_raw"]

        self.labels_syn = kwargs["labels_syn"]

        self.proposals_feat: List[Feature] = kwargs["proposals_feat"]
        """A list of lists of floats representing the proposals' features"""

        self.targets: List[Box] = kwargs["targets"]
        """A list of bounding boxes, representing the target for each query"""

        self.locations = kwargs["locations"]

        self.relations = kwargs["relations"]

    def __str__(self):
        return f"ReferitSample(image_id={self.image_id}, ann_id={self.ann_id})"


class ReferitDatum:
    def __init__(self, image_id, ann_id, *, precomputed, nlp=None):
        self.image_id = image_id
        self.ann_id = ann_id
        self.identifier = ann_id  # alias
        self.precomputed = precomputed
        self.nlp = nlp

    def as_samples(self) -> List[ReferitSample]:
        sample = {
            # meta
            "image_id": self.image_id,
            "ann_id": self.ann_id,
            # text
            "sentence": self.get_sentence(),
            "queries": self.get_queries(),
            "heads": self.get_heads(),
            # image
            "image_w": self.get_image_w(),
            "image_h": self.get_image_h(),
            # box
            "proposals": self.get_proposals(),
            "labels": self.get_labels(),
            "attrs": self.get_attrs(),
            "labels_syn": self.get_labels_syn(),
            "labels_raw": self.get_labels_raw(),
            # feats
            "proposals_feat": self.get_feats(),
            # targets
            "targets": self.get_targets(),
            # relations
            "locations": self.get_locations(),
            "relations": self.get_relations(),
        }

        return [ReferitSample(**sample)]

    def get_sentence(self):
        # referit has one query for almost all examples, in case of multiple queries
        # we just return the first one as the sentence
        return self.get_queries()[0]

    def get_queries(self):
        return self._pre("refs").get_queries(self.ann_id)

    def get_heads(self):
        if self._has_precomputed("heads"):
            return self._pre("heads").get_heads(self.ann_id)

        def get_head(cs) -> str:
            heads = [c.root.text for c in cs]
            head = " ".join(heads)
            return head

        queries = self.get_queries()

        docs = [self.nlp(query) for query in queries]
        chunks = [doc.noun_chunks for doc in docs]
        heads = [get_head(cs) for cs in chunks]

        fallbacks = [doc[-1].text for doc in docs]

        heads = [head if head else fallback for head, fallback in zip(heads, fallbacks)]

        return heads

    def get_image_w(self):
        return self._pre("images_size").get_width(self.image_id)

    def get_image_h(self):
        return self._pre("images_size").get_height(self.image_id)

    def get_proposals(self):
        return self._pre("objects_detection").get_boxes(self.image_id)

    def get_labels(self):
        return self._pre("objects_detection").get_classes(self.image_id)

    def get_attrs(self):
        return self._pre("objects_detection").get_attrs(self.image_id)

    def get_labels_syn(self):
        return [
            [label] for label in self.get_labels()
        ]  # TODO: temporary workaround: setting the label itself as alternative

    def get_labels_raw(self):
        return self.get_labels()  # TODO: temporary workaround

    def get_feats(self):
        return self.precomputed["objects_feature"].get_feature(self.image_id)

    def get_targets(self):
        x, y, w, h = self._pre("annotations").get_target(self.ann_id)
        return [[x, y, x + w, y + h] for _ in range(len(self.get_queries()))]

    def get_locations(self):
        return get_locations(self.get_queries())

    def get_relations(self):
        proposals = self.get_proposals()
        labels = self.get_labels()

        return get_relations(proposals, labels)

    def __str__(self):
        return f"ReferitDatum(image_id={self.image_id}, ann_id={self.ann_id})"

    def _pre(self, key):
        try:
            return self.precomputed[key]
        except KeyError:
            raise NotImplementedError(
                f"Extracting {key} is not supported, please provide precomputed data"
            )

    def _has_precomputed(self, key):
        return key in self.precomputed and self.precomputed[key] is not None


class ReferitDataset(Dataset, UpperboundAccuracyMixin, QueryNumberMixin):
    def __init__(
        self,
        split,
        data_dir,
        tokenizer,
        vocab,
        nlp=None,
        transform=None,
        identifiers=None,
    ):
        self.data_dir = data_dir
        self.split = split
        self.split_by = "berkeley"
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.nlp = nlp
        self.transform = transform

        self.identifiers = identifiers
        """A list identifiers for each image-sentence pair"""

        self.data: List[ReferitDatum] = None
        """A list of datum for each image-sentence pair"""

        self.samples: List[ReferitSample] = None
        """A list of batch-able samples for the model"""

        self.load()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # ann_id can't be part of meta because it is a string and
        # can't be represented in a tensor
        meta = [idx, sample.image_id]

        item = {
            "meta": meta,
            "sentence": self._prepare_sentence(sample.sentence),
            "queries": self._prepare_queries(sample.queries),
            "heads": self._prepare_queries(sample.heads),
            "image_w": sample.image_w,
            "image_h": sample.image_h,
            "proposals": sample.proposals,
            "labels": self._prepare_labels(sample.labels),
            "attrs": self._prepare_labels(sample.attrs),
            "labels_raw": self._prepare_labels(sample.labels_raw),
            "labels_syn": self._prepare_labels_syn(sample.labels_syn),
            "proposals_feat": sample.proposals_feat,
            "targets": sample.targets,
            "locations": sample.locations,
            "relations": sample.relations,
        }

        if self.transform:
            item = self.transform(item)

        return item

    def load(self):
        self._load_identifiers()

        precomputed = {
            "refs": self._open_refs_repo(),
            "annotations": self._open_annotations_repo(),
            "heads": self._open_heads_repo(),
            "images_size": self._open_images_size_repo(),
            "objects_detection": self._open_objects_detection_repo(),
            "objects_feature": self._open_objects_feature_repo(),
        }

        data = []
        samples = []

        logging.info(f"Loading {len(self.identifiers)} image-sentence pairs")

        for image_id, ann_id in self.identifiers:
            datum = ReferitDatum(
                image_id, ann_id, precomputed=precomputed, nlp=self.nlp
            )

            data += [datum]
            samples += datum.as_samples()

        self.data = data
        self.samples = samples

    def print_statistics(self):
        print(f"ReferIt ({self.split})")
        print(f"Number of images-sentences pairs: {len(self.data)}")
        print(f"Number of samples: {len(self)}")
        print(f"Upperbound accuracy: {self.get_upperbound_accuracy() * 100:.2f}%")
        print(f"Min number of queries: {self.get_min_query_number()}")
        print(f"Max number of queries: {self.get_max_query_number()}")
        print(f"Avg number of queries: {self.get_avg_query_number():.2f}")

    def get_image_path(self, image_id):
        image_id_str = str(image_id)
        image_id_str = image_id_str.zfill(5)

        image_id_part1 = image_id_str[:2]

        return f"{self.data_dir}/refer/data/images/saiapr_tc-12/{image_id_part1}/images/{image_id}.jpg"

    def _load_identifiers(self):
        """
        Load the identifiers given the pickled split file.
        Some images are blacklisted because they are missing in the features file.

        The split file is a pickled list dict like
        ```
        {
            "sent_ids": [130363],
            "ann_id": "7380_6",
            "ref_id": 99295,
            "image_id": 7380,
            "split": "test",
            "sentences": [{"tokens": ["sky"], "raw": "sky", "sent_id": 130363, "sent": "sky"}],
            "category_id": 224,
        }
        ```
        """
        if self.identifiers:
            return  # already loaded

        refs_repo = self._open_refs_repo()

        self.identifiers = refs_repo.get_identifiers(self.split)
        self.identifiers = [
            (image_id, ann_id)
            for image_id, ann_id in self.identifiers
            if not self._is_blacklisted(image_id)
        ]

    def _prepare_sentence(self, sentence: str) -> List[int]:
        sentence = self.tokenizer(sentence)
        sentence = self.vocab(sentence)
        return sentence

    def _prepare_queries(self, queries: List[str]) -> List[List[int]]:
        queries = [self.tokenizer(query) for query in queries]
        queries = [self.vocab(query) for query in queries]
        return queries

    def _prepare_labels(self, labels: List[str]) -> List[int]:
        return self.vocab(labels)

    def _prepare_labels_syn(self, labels: List[List[str]]) -> List[List[int]]:
        return [self.vocab(alternatives) for alternatives in labels]

    def _open_refs_repo(self):
        refs_file = f"{self.data_dir}/refer/data/refclef/refs({self.split_by}).p"
        return RefsRepository(refs_file)

    def _open_annotations_repo(self):
        annotations_file = f"{self.data_dir}/refer/data/refclef/instances.json"
        return AnnotationsRepository(annotations_file)

    def _open_images_size_repo(self):
        images_size_file = os.path.join(self.data_dir, self.split + "_images_size.json")
        return ImagesSizeRepository(images_size_file)

    def _open_objects_detection_repo(self):
        objects_detection_file = os.path.join(
            self.data_dir, self.split + "_detection_dict.json"
        )
        return ObjectsDetectionRepository(objects_detection_file)

    def _open_objects_feature_repo(self):
        id2idx_path = os.path.join(self.data_dir, self.split + "_imgid2idx.pkl")
        objects_feature_file = os.path.join(
            self.data_dir, self.split + "_features_compress.hdf5"
        )
        return ObjectsFeature2Repository(id2idx_path, objects_feature_file)

    def _open_heads_repo(self):
        heads_file = os.path.join(self.data_dir, self.split + "_heads.json")
        if not os.path.exists(heads_file):
            return None
        return HeadsRepository(heads_file)

    def _is_blacklisted(self, image_id):
        # some image ids are missing in the features file for some reason,
        # ignoring them

        blacklist_test = [
            37325,
            37326,
            30840,
            6897,
            8876,
            2972,
        ]
        blacklist_train = [
            2976,
        ]

        blacklist = blacklist_test + blacklist_train

        return image_id in blacklist


if __name__ == "__main__":
    # Run with: python -m weakvg.referit

    import logging
    import argparse

    from weakvg.wordvec import get_nlp, get_tokenizer, get_wordvec

    logging.basicConfig(level=logging.DEBUG)

    args = argparse.ArgumentParser(description="Load ReferIt Dataset")
    args.add_argument("--split", type=str, default="val")

    args = args.parse_args()

    split = args.split

    tokenizer = get_tokenizer()
    nlp = get_nlp()
    wordvec, vocab = get_wordvec()

    referit = ReferitDataset(split, "data/referit", tokenizer, vocab, nlp)

    referit.print_statistics()
