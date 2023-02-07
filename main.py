import logging
import pytorch_lightning as pl
from weakvg.dataset import Flickr30kDataModule
from weakvg.wordindexer import WordIndexer

# import wandb
import torch
from typing import Any
import torchtext

class MyModel(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(32, 2)

    def forward(self, x):
        return x

    # def training_step(self, batch, batch_idx):
    #     # inputs
    #     sentence = batch["sentence"]
    #     query = batch["query"]
    #     labels = batch["labels"]
    #     proposals = batch["proposals"]
    #     proposals_feat = batch["proposals_feat"]

    #     y_hat = self(x)
    #     return loss

    def training_step(self, batch, batch_idx):
        return batch

    def validation_step(self, batch, batch_idx):
        return batch

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pl.seed_everything(42)

    # wandb.init(project="weakvg", entity="weakvg")

    # wandb_logger = pl.loggers.WandbLogger(
    #     project="weakvg",
    #     entity="weakvg",
    #     log_model=True,
    #     save_dir="data/wandb",
    # )

    wordvec = torchtext.vocab.GloVe(name='6B', dim=300, cache="data/glove")

    pad_token = '<PAD>'
    unk_token = '<UNK>'
    vocab = torchtext.vocab.vocab(wordvec.stoi, specals=[pad_token, unk_token], default_index=-1)


    word_indexer = WordIndexer()

    dm = Flickr30kDataModule(
        data_dir="data/flickr30k",
        word_indexer=word_indexer,
        batch_size=32,
        num_workers=4,
        train_fraction=1.0,
    )

    model = MyModel()

    trainer = pl.Trainer(
        gpus=0,
        max_epochs=25,
    )

    trainer.fit(model, dm)
