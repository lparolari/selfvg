import json
import logging

import pytorch_lightning as pl
from torchtext.data.utils import get_tokenizer

from weakvg.cli import get_args, get_callbacks, get_logger
from weakvg.dataset import Flickr30kDataModule
from weakvg.model import MyModel
from weakvg.wordvec import get_nlp, get_objects_vocab, get_wordvec


def main():
    args = get_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logging.info("Args: " + json.dumps(vars(args), indent=4))

    logger = get_logger(args)
    callbacks = get_callbacks(args)

    pl.seed_everything(42, workers=True)

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

    model = MyModel(wordvec, vocab, omega=args.omega, task=args.task)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
    )

    if args.mode == "train":
        trainer.fit(model, dm)

    if args.mode == "test":
        trainer.test(model, dm)


if __name__ == "__main__":
    main()
