import os
import re
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from weakvg.padding import (
    pad_labels,
    pad_proposals,
    pad_queries,
    pad_sentence,
    pad_targets,
)
from weakvg.repo import (
    ImagesSizeRepository,
    ObjectsDetectionRepository,
    ObjectsFeatureRepository,
)

Box = List[int]


class Flickr30kDatum:
    def __init__(
        self,
        identifier: int,
        *,
        data_dir: str,
        precomputed: Dict[str, Dict[int, Any]],
        nlp=None,
    ):
        self.identifier = identifier
        self.data_dir = data_dir
        self.precomputed = precomputed
        self.nlp = nlp

        self._sentences_ann = None
        self._targets_ann = None

        self._load_sentences_ann()
        self._load_targets_ann()

    def get_sentences_ann(self) -> List[str]:
        return self._sentences_ann

    def get_sentences_ids(self) -> List[int]:
        return list(range(len(self.get_sentences_ann())))

    def get_sentence(self, sentence_id, *, return_ann=False) -> str:
        sentence_ann = self._sentences_ann[sentence_id]
        sentence = self._remove_sentence_ann(sentence_ann)
        sentence = self._process_text(sentence)

        if return_ann:
            return sentence, sentence_ann

        return sentence

    def get_queries_ids(self, sentence_id) -> List[int]:
        return list(range(len(self.get_queries(sentence_id))))

    def get_queries(self, sentence_id, query_id=None, *, return_ann=False) -> List[str]:
        _, sentence_ann = self.get_sentence(sentence_id, return_ann=True)

        queries_ann = self._extract_queries_ann(sentence_ann)

        # important: we need to filter out queries without targets in order to
        #            produce aligned data
        queries_ann = [
            query_ann for query_ann in queries_ann if self.has_target_for(query_ann)
        ]

        queries = [self._extract_phrase(query_ann) for query_ann in queries_ann]
        queries = [self._process_text(query) for query in queries]

        a_slice = slice(query_id, query_id and query_id + 1)

        if return_ann:
            return queries[a_slice], queries_ann[a_slice]

        return queries[a_slice]

    def get_heads(self, sentence_id, query_id=None) -> List[str]:
        a_slice = slice(query_id, query_id and query_id + 1)
        queries = self.get_queries(sentence_id, query_id)

        if not self.nlp:
            # as fallback we return the full query, which, instead
            # of being a specific part of the query is the query itself
            return queries

        def get_head(cs) -> str:
            heads = [c.root.text for c in cs]
            head = " ".join(heads)
            return head

        docs = [self.nlp(query) for query in queries]
        chunks = [doc.noun_chunks for doc in docs]
        heads = [get_head(cs) for cs in chunks]

        fallbacks = [doc[-1].text for doc in docs]

        heads = [head if head else fallback for head, fallback in zip(heads, fallbacks)]

        return heads[a_slice]

    def get_targets(self, sentence_id) -> List[List[int]]:
        targets_ann = self._targets_ann
        _, queries_ann = self.get_queries(sentence_id, return_ann=True)

        return [
            targets_ann[self._get_entity_id(query_ann)] for query_ann in queries_ann
        ]

    def get_image_w(self) -> int:
        if "images_size" not in self.precomputed:
            raise NotImplementedError(
                "Extracting image width is not supported, please provide precomputed data"
            )
        return self.precomputed["images_size"].get_width(self.identifier)

    def get_image_h(self) -> int:
        return self.precomputed["images_size"].get_height(self.identifier)

    def get_proposals(self) -> List[List[int]]:
        return self.precomputed["objects_detection"].get_boxes(self.identifier)

    def get_classes(self) -> List[str]:
        return self.precomputed["objects_detection"].get_classes(self.identifier)

    def get_attrs(self) -> List[str]:
        return self.precomputed["objects_detection"].get_attrs(self.identifier)

    def get_proposals_feat(self) -> np.array:
        return self.precomputed["objects_feature"].get_feature(
            self.identifier
        )  # [x, 2048]

    def has_queries_for(self, sentence_id) -> bool:
        return len(self.get_queries(sentence_id)) > 0

    def has_target_for(self, query_ann: str) -> bool:
        return self._get_entity_id(query_ann) in self._targets_ann

    def __iter__(self):
        sentence_ids = self.get_sentences_ids()

        for sentence_id in sentence_ids:
            if not self.has_queries_for(sentence_id):
                continue

            yield {
                "identifier": self.identifier,
                # text
                "sentence": self.get_sentence(sentence_id),
                "queries": self.get_queries(sentence_id),
                "heads": self.get_heads(sentence_id),
                # image
                "image_w": self.get_image_w(),
                "image_h": self.get_image_h(),
                # box
                "proposals": self.get_proposals(),
                "labels": self.get_classes(),
                "attrs": self.get_attrs(),
                # feats
                "proposals_feat": self.get_proposals_feat(),
                # targets
                "targets": self.get_targets(sentence_id),
            }

    def _get_sentences_file(self) -> str:
        return os.path.join(
            self.data_dir, "Flickr30kEntities", "Sentences", f"{self.identifier}.txt"
        )

    def _load_sentences_ann(self) -> List[str]:
        with open(self._get_sentences_file(), "r") as f:
            sentences = [x.strip() for x in f]

        self._sentences_ann = sentences

    def _load_targets_ann(self):
        from xml.etree.ElementTree import parse

        targets_ann = {}

        annotation_file = os.path.join(
            self.data_dir, "Flickr30kEntities", "Annotations", f"{self.identifier}.xml"
        )

        root = parse(annotation_file).getroot()
        elements = root.findall("./object")

        for element in elements:
            bndbox = element.find("bndbox")

            if bndbox is None or len(bndbox) == 0:
                continue

            left = int(element.findtext("./bndbox/xmin"))
            top = int(element.findtext("./bndbox/ymin"))
            right = int(element.findtext("./bndbox/xmax"))
            bottom = int(element.findtext("./bndbox/ymax"))

            for name in element.findall("name"):
                entity_id = int(name.text)

                if not entity_id in targets_ann.keys():
                    targets_ann[entity_id] = []
                targets_ann[entity_id].append([left, top, right, bottom])

        self._targets_ann = targets_ann

    @staticmethod
    def _remove_sentence_ann(sentence: str) -> str:
        return re.sub(r"\[[^ ]+ ", "", sentence).replace("]", "")

    @staticmethod
    def _extract_queries_ann(sentence_ann: str) -> List[str]:
        query_pattern = r"\[(.*?)\]"
        queries_ann = re.findall(query_pattern, sentence_ann)
        return queries_ann

    @staticmethod
    def _extract_phrase(query_ann: str) -> str:
        """
        Extracts the phrase from the query annotation

        Example:
          '/EN#283585/people A young white boy' --> 'A young white boy'
        """
        return query_ann.split(" ", 1)[1]

    @staticmethod
    def _extract_entity(query_ann: str) -> str:
        """
        Extracts the entity from the query annotation

        Example:
          '/EN#283585/people A young white boy' --> '/EN#283585/people'
        """
        return query_ann.split(" ", 1)[0]

    @staticmethod
    def _get_entity_id(query_ann: str) -> int:
        """
        Extracts the entity id from the query annotation

        Example:
          '/EN#283585/people A young white boy' --> 283585
        """
        entity = Flickr30kDatum._extract_entity(query_ann)

        entity_id_pattern = r"\/EN\#(\d+)"

        matches = re.findall(entity_id_pattern, entity)
        first_match = matches[0]

        return int(first_match)

    @staticmethod
    def _process_text(text: str) -> str:
        return text.lower()


class Flickr30kDataset(Dataset):
    def __init__(self, split, data_dir, tokenizer, vocab, nlp=None, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.nlp = nlp
        self.transform = transform

        self.identifiers = None
        """A list identifiers for each image-sentence pair"""

        self.data = None
        """A list of image-sentence pair datum"""

        self.samples = None
        """A list of batch-able samples for the model"""

        self.load()
        self.preflight_check()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        meta = [idx, sample["identifier"]]
        sentence = self._prepare_sentence(sample["sentence"])
        queries = self._prepare_queries(sample["queries"])
        heads = self._prepare_queries(sample["heads"])
        image_w = sample["image_w"]
        image_h = sample["image_h"]
        proposals = sample["proposals"]
        labels = self._prepare_labels(sample["labels"])
        attrs = self._prepare_labels(sample["attrs"])
        proposals_feat = sample["proposals_feat"]
        targets = self._prepare_targets(sample["targets"])

        assert len(queries) == len(
            targets
        ), f"Expected length of `targets` to be {len(queries)}, got {len(targets)}"
        assert len(proposals) == len(
            labels
        ), f"Expected length of `labels` to be {len(proposals)}, got {len(labels)}"
        assert len(proposals) == len(
            attrs
        ), f"Expected length of `attrs` to be {len(proposals)}, got {len(attrs)}"
        assert len(proposals) == len(
            proposals_feat
        ), f"Expected length of `proposals_feat` to be {len(proposals)}, got {len(proposals_feat)}"

        item = {
            "meta": meta,
            "sentence": sentence,
            "queries": queries,
            "heads": heads,
            "image_w": image_w,
            "image_h": image_h,
            "proposals": proposals,
            "labels": labels,
            "attrs": attrs,
            "proposals_feat": proposals_feat,
            "targets": targets,
        }

        if self.transform:
            item = self.transform(item)

        return item

    def load(self):
        self._load_identifiers()

        images_size_repo = self._open_images_size()
        objects_detection_repo = self._open_objects_detection()
        objects_feature_repo = self._open_objects_feature()

        precomputed = {
            "images_size": images_size_repo,
            "objects_detection": objects_detection_repo,
            "objects_feature": objects_feature_repo,
        }

        samples = []
        data = []

        logging.debug(f"Loading {len(self.identifiers)} images...")

        for identifier in self.identifiers:
            datum = Flickr30kDatum(
                identifier,
                data_dir=self.data_dir,
                precomputed=precomputed,
                nlp=self.nlp,
            )

            data.append(datum)

            for sample in datum:
                samples.append(sample)

        self.samples = samples
        self.data = data

    def preflight_check(self):
        if len(self.identifiers) == 0:
            raise RuntimeError("Empty dataset, please check the identifiers file")

        if len(self.samples) == 0:
            raise RuntimeError("Cannot create dataset with 0 samples")

    def get_split(self):
        return self.split

    def get_upperbound_accuracy(self):
        total = 0
        matched = 0

        for sample in iter(self):
            targets = sample["targets"]
            proposals = sample["proposals"]

            for target in targets:
                iou_scores = [iou(target, proposal) for proposal in proposals]
                max_iou = max(iou_scores)

                if max_iou >= 0.5:
                    matched += 1

                total += 1

        return matched / total

    def print_statistics(self):
        print(f"Flickr30k ({self.split})")
        print(f"Number of images-sentences pairs: {len(self.data)}")
        print(f"Number of samples: {len(self)}")
        print(f"Upperbound accuracy: {self.get_upperbound_accuracy() * 100:.2f}")

    def _prepare_sentence(self, sentence: str) -> List[int]:
        sentence = self.tokenizer(sentence)
        sentence = self.vocab(sentence)
        return sentence

    def _prepare_queries(self, queries: List[str]) -> List[List[int]]:
        queries = [self.tokenizer(query) for query in queries]
        queries = [self.vocab(query) for query in queries]
        return queries

    def _prepare_labels(self, labels: List[str]) -> List[List[int]]:
        return self.vocab(labels)

    def _prepare_targets(self, targets: List[List[Box]]) -> List[Box]:
        return [union_box(target) for target in targets]

    def _load_identifiers(self):
        identifier_file = os.path.join(
            self.data_dir, "Flickr30kEntities", self.split + ".txt"
        )

        with open(identifier_file, "r") as f:
            identifiers = f.readlines()

        identifiers = [int(identifier.strip()) for identifier in identifiers]

        self.identifiers = identifiers

    def _open_images_size(self):
        images_size_file = os.path.join(self.data_dir, self.split + "_images_size.json")
        return ImagesSizeRepository(images_size_file)

    def _open_objects_detection(self):
        objects_detection_file = os.path.join(
            self.data_dir, self.split + "_detection_dict.json"
        )
        return ObjectsDetectionRepository(objects_detection_file)

    def _open_objects_feature(self):
        id2idx_path = os.path.join(self.data_dir, self.split + "_imgid2idx.pkl")
        objects_feature_file = os.path.join(
            self.data_dir, self.split + "_features_compress.hdf5"
        )
        return ObjectsFeatureRepository(id2idx_path, objects_feature_file)


class Flickr30kDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        tokenizer,
        vocab,
        nlp,
        num_workers=1,
        batch_size=32,
        train_fraction=1,
        dev=False,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_fraction = train_fraction
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.nlp = nlp
        self.dev = dev

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # we assume dataset is already downloaded
        pass

    def setup(self, stage=None):
        dataset_kwargs = {
            "data_dir": self.data_dir,
            "tokenizer": self.tokenizer,
            "vocab": self.vocab,
            "nlp": self.nlp,
            "transform": NormalizeCoord(),
        }

        if stage == "fit" or stage is None:
            self.train_dataset = Flickr30kDataset(
                split="val" if self.dev else "train", **dataset_kwargs
            )
            self.val_dataset = Flickr30kDataset(split="val", **dataset_kwargs)

        if stage == "test" or stage is None:
            self.test_dataset = Flickr30kDataset(split="test", **dataset_kwargs)

    def train_dataloader(self):
        from torch.utils.data.sampler import SubsetRandomSampler

        n_samples = len(self.train_dataset)
        n_subset = int(n_samples * self.train_fraction)

        train_indices = np.random.choice(n_samples, size=n_subset, replace=False)

        sampler = SubsetRandomSampler(train_indices)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=sampler,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )


def collate_fn(batch):
    sentence_max_length = 32
    query_max_length = 12
    head_max_length = 5
    proposal_max_length = 100

    batch = pd.DataFrame(batch).to_dict(orient="list")

    return {
        "meta": torch.tensor(batch["meta"]),
        "sentence": pad_sentence(batch["sentence"], sentence_max_length).long(),
        "queries": pad_queries(batch["queries"], query_max_length).long(),
        "heads": pad_queries(batch["heads"], head_max_length).long(),
        "image_w": torch.tensor(batch["image_w"]),
        "image_h": torch.tensor(batch["image_h"]),
        "proposals": pad_proposals(batch["proposals"], proposal_max_length).float(),
        "labels": pad_labels(batch["labels"], proposal_max_length),
        "attrs": pad_labels(batch["attrs"], proposal_max_length),
        "proposals_feat": pad_proposals(batch["proposals_feat"], proposal_max_length),
        "targets": pad_targets(batch["targets"]).float(),
    }


class NormalizeCoord:
    def __call__(self, sample):
        w, h = sample["image_w"], sample["image_h"]

        size = np.array([w, h, w, h])

        proposals = np.array(sample["proposals"]).astype(np.float64)
        targets = np.array(sample["targets"]).astype(np.float64)

        proposals_norm = proposals / size
        targets_norm = targets / size

        return {
            **sample,
            "proposals": proposals_norm,
            "targets": targets_norm,
        }


def union_box(boxes: List[Box]):
    x1 = min([box[0] for box in boxes])
    y1 = min([box[1] for box in boxes])
    x2 = max([box[2] for box in boxes])
    y2 = max([box[3] for box in boxes])
    return [x1, y1, x2, y2]


def iou(box_a: Box, box_b: Box):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union


if __name__ == "__main__":
    # Run with: python -m weakvg.dataset

    import logging
    import argparse

    from weakvg.wordvec import get_tokenizer, get_wordvec, get_nlp, get_objects_vocab

    logging.basicConfig(level=logging.DEBUG)

    args = argparse.ArgumentParser(
        description="Load Flickr30k Dataset and print its statistics"
    )
    args.add_argument("--split", type=str, default="val")
    args.add_argument("--load_objects_vocab", action="store_true", default=False)

    args = args.parse_args()

    load_objects_vocab = args.load_objects_vocab
    split = args.split

    tokenizer = get_tokenizer()
    nlp = get_nlp()
    wordvec, vocab = get_wordvec(
        custom_tokens=get_objects_vocab() if load_objects_vocab else []
    )

    ds = Flickr30kDataset(
        split=split,
        data_dir="data/flickr30k",
        tokenizer=tokenizer,
        vocab=vocab,
        nlp=nlp,
    )

    ds.print_statistics()
