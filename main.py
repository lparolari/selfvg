import logging

import pytorch_lightning as pl
from torchtext.data.utils import get_tokenizer

from weakvg.dataset import Flickr30kDataModule
from weakvg.model import MyModel
from weakvg.wordvec import get_wordvec

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

    tokenizer = get_tokenizer("basic_english")
    wordvec, vocab = get_wordvec()

    dm = Flickr30kDataModule(
        data_dir="data/flickr30k",
        batch_size=4,
        num_workers=1,
        train_fraction=1.0,
        tokenizer=tokenizer,
        vocab=vocab,
    )

    model = MyModel(wordvec, vocab)

    trainer = pl.Trainer(
        gpus=0,
        max_epochs=25,
    )

    trainer.fit(model, dm)
