import json
import logging
import pickle
from typing import List, Dict, Any

import h5py
import numpy as np


class ImagesSizeRepository:
    def __init__(self, images_size_path) -> None:
        logging.debug(f"Loading images size at {images_size_path}...")
        with open(images_size_path, "r") as f:
            self.images_size = json.load(f)

    def get_width(self, identifier):
        return self.images_size[str(identifier)][0]

    def get_height(self, identifier):
        return self.images_size[str(identifier)][1]


class ObjectsDetectionRepository:
    def __init__(self, objects_detection_path) -> None:
        logging.debug(f"Loading objects detection at {objects_detection_path}...")
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
        logging.debug(f"Loading imgid2idx at {id2idx_path}...")

        with open(id2idx_path, "rb") as f:
            self.id2idx = pickle.load(f)

        logging.debug(f"Loading objects feature at {objects_feature_path}...")

        with h5py.File(objects_feature_path, "r") as hf:
            # print(hf.keys()) 	#<KeysViewHDF5 ['bboxes', 'features', 'pos_bboxes']>
            self.features = np.array(hf.get("features"))
            self.positions = np.array(hf.get("pos_bboxes"))

    def get_feature(self, identifier):
        idx = self.id2idx[identifier]
        pos = self.positions[idx]
        return self.features[pos[0] : pos[1]]


class ObjectsFeature2Repository:
    def __init__(self, id2idx_path, objects_feature_path) -> None:
        logging.debug(f"Loading imgid2idx at {id2idx_path}...")

        with open(id2idx_path, "rb") as f:
            self.id2idx = pickle.load(f)

        logging.debug(f"Loading objects feature at {objects_feature_path}...")

        with h5py.File(objects_feature_path, "r") as hf:
            # print(hf.keys()) 	#<KeysViewHDF5 ['image_bb', 'image_features', 'pos_boxes', 'spatial_features']>
            self.features = np.array(hf.get("image_features"))
            self.positions = np.array(hf.get("pos_boxes"))

    def get_feature(self, identifier):
        idx = self.id2idx[identifier]
        pos = self.positions[idx]
        return self.features[pos[0] : pos[1]]


class HeadsRepository:
    def __init__(self, heads_path) -> None:
        logging.debug(f"Loading heads at {heads_path}...")
        with open(heads_path, "r") as f:
            self.heads = json.load(f)

    def get_heads(self, identifier, sentence_id=None):
        heads = self.heads[str(identifier)]

        if sentence_id is not None:
            return heads[sentence_id]

        return heads


class LabelsRepository:
    """
    Note: indexes encoded in alternatives are 1-based
    """

    def __init__(self, labels, alternatives):
        self.labels = labels
        self.alternatives = alternatives

        self.idx2labels = None
        self.labels2idx = None
        self.idx2alternatives = None
        self.alternatives2idx = None
        self.labelidx2alternativesidx = None

        self._build_tables()
        self._build_mappings()

    @classmethod
    def from_vocab(cls, labels_path, alternatives_path):
        labels = cls.read_labels(labels_path)
        alternatives = cls.read_alternatives(alternatives_path)

        return cls(labels, alternatives)

    def get_alternatives(self, label: str) -> List[str]:
        labelidx = self.labels2idx[label]
        alternativesidx = self.labelidx2alternativesidx[labelidx]
        alternatives = self.idx2alternatives[alternativesidx]
        return [alternative.split(":")[1] for alternative in alternatives.split(",")]

    def get_raw(self, label: str) -> str:
        labelidx = self.labels2idx[label]
        alternativesidx = self.labelidx2alternativesidx[labelidx]
        alternatives = self.idx2alternatives[alternativesidx]
        return alternatives

    @classmethod
    def read_labels(cls, labels_path):
        logging.debug(f"Loading labels at {labels_path}...")

        with open(labels_path, "r") as f:
            labels = f.readlines()
            labels = [label.strip("\n").strip() for label in labels]

        return labels

    @classmethod
    def read_alternatives(cls, alternatives_path):
        logging.debug(f"Loading alternatives at {alternatives_path}...")

        with open(alternatives_path, "r") as f:
            alternatives = f.readlines()
            alternatives = [
                alternative.strip("\n").strip() for alternative in alternatives
            ]

        return alternatives

    def _build_tables(self):
        self.labels2idx = {label: i + 1 for i, label in enumerate(self.labels)}
        self.idx2labels = {i + 1: label for i, label in enumerate(self.labels)}
        self.alternatives2idx = {
            alternative: i + 1 for i, alternative in enumerate(self.alternatives)
        }
        self.idx2alternatives = {
            i + 1: alternative for i, alternative in enumerate(self.alternatives)
        }

    def _build_mappings(self):
        self.labelidx2alternativesidx, _, _ = self._get_categories_mapping()

    def _get_categories_mapping(self):
        # NOTE: This code is taken as is from Davide Rigoni's repo

        cleaned_labels = self.idx2alternatives

        map_fn = dict()
        old_labels = dict()
        for new_label_id, new_label_str in cleaned_labels.items():
            new_label_id = int(new_label_id)
            for piece in new_label_str.split(","):
                tmp = piece.split(":")
                assert len(tmp) == 2
                old_label_id = int(tmp[0])
                old_label_str = tmp[1]
                # we need to avoid overriding of same ids like: 17:stop sign,17:stopsign
                if old_label_id not in old_labels.keys():
                    old_labels[old_label_id] = old_label_str
                    map_fn[old_label_id] = new_label_id
                else:
                    old_labels[old_label_id] = (
                        old_labels[old_label_id] + "," + old_label_str
                    )
                    # print(f"--{old_labels[old_label_id]}--")
                    # print('Warning: label already present for {}:{}. Class {} ignored. '.format(old_label_id,
                    #                                                                             old_labels[old_label_id],
                    #                                                                             old_label_str))
        # assert len(old_labels) == 1600
        # assert len(old_labels) == len(map_fn)
        # print(old_labels[1590], map_fn[1590], cleaned_labels[map_fn[1590]])
        return map_fn, cleaned_labels, old_labels  # all in [1, 1600]


class AnnotationsRepository:
    def __init__(self, annotations_file):
        self.annotations_file = annotations_file

        self.annotations: List[Dict[str, Any]] = []
        """An array of annotations"""

        self.annid2idx: Dict[str, int] = {}
        """A dictionary that maps `ann_id` to its corresponding index in `self.annotations`"""

        self._load()
        self._build_annid2idx()

    def get_target(self, ann_id):
        idx = self.annid2idx[ann_id]
        return self.annotations[idx]["bbox"]

    def _load(self):
        with open(self.annotations_file) as f:
            data = json.load(f)
            self.annotations = data["annotations"]

    def _build_annid2idx(self):
        for i, ann in enumerate(self.annotations):
            self.annid2idx[ann["id"]] = i


class RefsRepository:
    def __init__(self, refs_file):
        self.refs_file = refs_file

        self.annid2idx: Dict[str, int] = {}
        self.data = None

        self._load()
        self._build_annid2idx()

    def get_identifiers(self, split):
        return [
            (ref["image_id"], ref["ann_id"])
            for ref in self.data
            if ref["split"] == split
        ]

    def get_queries(self, ann_id):
        ref = self._get_ref(ann_id)
        return [sentence["raw"].strip().lower() for sentence in ref["sentences"]]

    def get_target(self, ann_id):
        ref = self._get_ref(ann_id)
        return ref["bbox"]

    def _get_ref(self, ann_id):
        idx = self.annid2idx[ann_id]
        ref = self.data[idx]
        return ref

    def _load(self):
        with open(self.refs_file, "rb") as f:
            self.data = pickle.load(f)

    def _build_annid2idx(self):
        for i, ref in enumerate(self.data):
            self.annid2idx[ref["ann_id"]] = i
