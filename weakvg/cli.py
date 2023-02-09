import argparse
import json
import logging

import pytorch_lightning as pl


def get_args(verbose=True):
    parser = argparse.ArgumentParser()

    # global params
    parser.add_argument("--exp_id", type=str, default="")
    parser.add_argument("--mode", type=str, choices=["train"], default="train")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Dev mode uses validation set instead of training set",
    )
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument(
        "--logger", type=str, choices=["wandb", "tensorboard"], default="tensorboard"
    )
    parser.add_argument("--max_epochs", type=int, default=25)

    # dataset params
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_fraction", type=float, default=1.0)

    args = parser.parse_args()

    if verbose:
        logging.info("Args: " + json.dumps(vars(args), indent=4))

    return args


def get_logger(args):
    name = args.exp_id or None

    if args.logger == "tensorboard":
        import os

        return pl.loggers.tensorboard.TensorBoardLogger(
            name=name, save_dir=os.path.join(os.getcwd(), "tb_logs")
        )

    if args.logger == "wandb":
        import wandb

        return pl.loggers.WandbLogger(
            project="weakly_guys",
            entity="weakvg",
            log_model=True,
            save_dir="output",
            name=name,
            settings=wandb.Settings(start_method="fork"),
        )

    return None
