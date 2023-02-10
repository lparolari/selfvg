import logging

import pytorch_lightning as pl
from torchtext.data.utils import get_tokenizer

from weakvg.dataset import Flickr30kDataModule
from weakvg.model import MyModel
from weakvg.wordvec import get_wordvec, get_objects_vocab, get_nlp
from weakvg.cli import get_args, get_logger, get_callbacks


def main():
    logging.basicConfig(level=logging.INFO)

    pl.seed_everything(42, workers=True)

    args = get_args()
    logger = get_logger(args)
    callbacks = get_callbacks(args)

    tokenizer = get_tokenizer("basic_english")
    wordvec, vocab = get_wordvec(custom_tokens=get_objects_vocab())
    nlp = get_nlp()

    dm = Flickr30kDataModule(
        data_dir="data/flickr30k",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_fraction=args.train_fraction,
        dev=args.dev,
        tokenizer=tokenizer,
        vocab=vocab,
        nlp=nlp,
    )

    model = MyModel(wordvec, vocab)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
