import json
import logging
import pickle

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
