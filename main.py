import json
import logging

import pytorch_lightning as pl

from weakvg.cli import get_args, get_callbacks, get_logger
from weakvg.dataset import Flickr30kDataModule
from weakvg.model import MyModel
from weakvg.wordvec import get_nlp, get_objects_vocab, get_tokenizer, get_wordvec


def main():
    args = get_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logging.info("Args: " + json.dumps(vars(args), indent=4))

    pl.seed_everything(42, workers=True)

    tokenizer = get_tokenizer()
    wordvec, vocab = get_wordvec(
        custom_labels=[] if args.dev else get_objects_vocab(),
        custom_tokens=get_objects_vocab("data/objects_vocab_merged.txt"),
    )
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

    if args.checkpoint:
        model = MyModel.load_from_checkpoint(
            args.checkpoint, wordvec=wordvec, vocab=vocab
        )
        logging.info(f"Loaded model at {args.checkpoint}")
    else:
        model = MyModel(
            wordvec,
            vocab,
            omega=args.omega,
            neg_selection=args.neg_selection,
        )

    logging.info(f"Model hparams: " + json.dumps(model.hparams_initial, indent=4))

    logger = get_logger(args, model)
    callbacks = get_callbacks(args)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
    )

    if "train" in args.mode:
        trainer.fit(model, dm)

    if "test" in args.mode:
        trainer.test(model, dm)


if __name__ == "__main__":
    main()
