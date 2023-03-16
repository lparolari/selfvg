import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from weakvg.dataset import Flickr30kDataset, ReferitDataset
from weakvg.padding import (
    pad_labels,
    pad_labels_syn,
    pad_proposals,
    pad_queries,
    pad_sentence,
    pad_targets,
)


class WeakvgDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
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

        if dataset not in ["flickr30k", "referit"]:
            raise ValueError(f"Unknown dataset {dataset}")

        self.dataset = dataset
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
        data_dir = WeakvgDataModule.get_data_dir(self.dataset)
        dataset_cls = WeakvgDataModule.get_dataset_cls(self.dataset)

        dataset_kwargs = {
            "data_dir": data_dir,
            "tokenizer": self.tokenizer,
            "vocab": self.vocab,
            "nlp": self.nlp,
            "transform": NormalizeCoord(),
        }

        if stage == "fit" or stage is None:
            self.train_dataset = dataset_cls(
                split="val" if self.dev else "train", **dataset_kwargs
            )
            self.val_dataset = dataset_cls(split="val", **dataset_kwargs)

        if stage == "test" or stage is None:
            self.test_dataset = dataset_cls(split="test", **dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=self._get_sampler(self.train_dataset, self.train_fraction),
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        sampler = (
            self._get_sampler(self.val_dataset, self.train_fraction)
            if self.dev
            else None
        )

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            sampler=sampler,
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

    def _get_sampler(self, dataset, fract):
        from torch.utils.data.sampler import SubsetRandomSampler

        n_samples = len(dataset)
        n_subset = int(n_samples * fract)

        train_indices = np.random.choice(n_samples, size=n_subset, replace=False)

        sampler = SubsetRandomSampler(train_indices)

        return sampler

    @classmethod
    def get_data_dir(cls, dataset: str):
        return {
            "flickr30k": "data/flickr30k",
            "referit": "data/referit",
        }[dataset]

    @classmethod
    def get_dataset_cls(cls, dataset: str):
        return {
            "flickr30k": Flickr30kDataset,
            "referit": ReferitDataset,
        }[dataset]


def collate_fn(batch):
    sentence_max_length = 32
    query_max_length = 12
    head_max_length = 5
    proposal_max_length = 100
    label_alternatives_max_length = 6

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
        "labels_raw": pad_labels(batch["labels_raw"], proposal_max_length),
        "labels_syn": pad_labels_syn(
            batch["labels_syn"], proposal_max_length, label_alternatives_max_length
        ),
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
