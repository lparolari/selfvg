import torch
import logging
import os
import pytorch_lightning as pl
import json
import pickle
import h5py
import numpy as np
import re

from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset


class Sample:
    def __init__(
        self, identifier: int, *, data_dir: str, precomputed: Dict[str, Dict[int, Any]]
    ):
        self.identifier = identifier
        self.data_dir = data_dir
        self.precomputed = precomputed

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

        if return_ann:
            return sentence, sentence_ann

        return sentence

    def get_queries_ids(self, sentence_id) -> List[int]:
        return list(range(len(self.get_queries(sentence_id))))

    def get_queries(self, sentence_id, query_id=None, *, return_ann=False) -> List[str]:
        _, sentence_ann = self.get_sentence(sentence_id, return_ann=True)

        a_slice = slice(query_id, query_id and query_id + 1)

        query_pattern = r"\[(.*?)\]"
        queries_ann = re.findall(query_pattern, sentence_ann)

        # query_ann has the entity annotation in the first part,
        # while query in the second
        # e.g.: '/EN#283585/people A young white boy'

        def get_phrase(query_ann):
            return query_ann.split(" ", 1)[1]

        queries = [get_phrase(query_ann) for query_ann in queries_ann]

        if return_ann:
            return queries[a_slice], queries_ann[a_slice]

        return queries[a_slice]

    def get_targets(self, sentence_id) -> List[List[int]]:
        _, queries_ann = self.get_queries(sentence_id, return_ann=True)

        entity_pattern = r"\/EN\#(\d+)"

        def get_ann(query_ann):
            return query_ann.split(" ", 1)[0]

        queries_ann = [get_ann(query_ann) for query_ann in queries_ann]
        entities = [int(re.findall(entity_pattern, ann)[0]) for ann in queries_ann]

        targets_ann = self._targets_ann

        targets = []

        for entity in entities:
            if not entity in targets_ann:
                continue

            targets.append(targets_ann[entity])

        return targets

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

    def as_entries(self) -> List[Dict[str, Any]]:
        entries = []

        for sentence_id in self.get_sentences_ann():
            entries.append(
                {
                    # text
                    "sentence": self.get_sentence(sentence_id),
                    "queries": self.get_queries(sentence_id),
                    # image
                    "image_w": self.get_image_w(),
                    "image_h": self.get_image_h(),
                    # box
                    "proposals": self.get_proposals(),
                    "classes": self.get_classes(),
                    "attrs": self.get_attrs(),
                    # feats
                    "proposals_feat": self.get_proposals_feat(),
                    # targets
                    "targets": self.get_targets(sentence_id),
                }
            )

        return entries

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

    def _remove_sentence_ann(self, sentence: str) -> str:
        return re.sub(r"\[[^ ]+ ", "", sentence).replace("]", "")


class Flickr30kDataset(Dataset):
    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.split = split

        self.identifiers = None
        """A list of numbers that identify each sample"""

        self.samples = None
        """A list of Sample objects"""

        self.load()
        self.preflight_check()

    def __len__(self):
        return len(self.identifiers)

    def __getitem__(self, idx):
        identifier = self.identifiers[idx]

        sample = self.samples[identifier]

        data = sample.as_entries()

        # sentence
        # queries

        # image_w
        # image_h

        # proposals
        # classes
        # attrs

        # proposals_feat

        # targets

        x = None
        y = data["targets"]

        return x, y

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

        for identifier in self.identifiers:
            sample = Sample(identifier, data_dir=self.data_dir, precomputed=precomputed)
            samples.append(sample)

        self.samples = samples

    def preflight_check(self):
        if len(self.identifiers) == 0:
            raise RuntimeError("Empty dataset, please check the identifiers file")

        if len(self.samples) != len(self.identifiers):
            raise RuntimeError(
                "The number of samples is different from the number of identifiers"
            )

    def _load_identifiers(self):
        identifier_file = os.path.join(
            self.data_dir, "Flickr30kEntities", self.split + ".txt"
        )

        with open(identifier_file, "r") as f:
            identifiers = f.readlines()

        identifiers = [int(identifier.strip()) for identifier in identifiers]

        self.identifiers = identifiers

    def _pad_targets(self, targets: List[int]) -> List[int]:
        x = [torch.tensor(target) for target in targets]
        return targets

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


class ImagesSizeRepository:
    def __init__(self, images_size_path) -> None:
        logging.info(f"Loading images size...")
        with open(images_size_path, "r") as f:
            self.images_size = json.load(f)

    def get_width(self, identifier):
        return self.images_size[str(identifier)][0]

    def get_height(self, identifier):
        return self.images_size[str(identifier)][1]


class ObjectsDetectionRepository:
    def __init__(self, objects_detection_path) -> None:
        logging.info(f"Loading objects detection...")
        with open(objects_detection_path, "r") as f:
            self.objects_detection = json.load(f)

    def get_boxes(self, identifier):
        return self.objects_detection[str(identifier)]["bboxes"]

    def get_classes(self, identifier):
        return self.objects_detection[str(identifier)]["classes"]

    def get_attrs(self, identifier):
        return self.objects_detection[str(identifier)]["attrs"]


class ObjectsFeatureRepository:
    def __init__(self, id2idx_path, objects_feature_path) -> None:
        logging.info(f"Loading objects feature img2id...")

        with open(id2idx_path, "rb") as f:
            self.id2idx = pickle.load(f)

        logging.info(f"Loading objects feature h5...")

        with h5py.File(objects_feature_path, "r") as hf:
            # print(hf.keys()) 	#<KeysViewHDF5 ['bboxes', 'features', 'pos_bboxes']>
            self.features = np.array(hf.get("features"))
            self.positions = np.array(hf.get("pos_bboxes"))

    def get_feature(self, identifier):
        idx = self.id2idx[identifier]
        pos = self.positions[idx]
        return self.features[pos[0] : pos[1]]
