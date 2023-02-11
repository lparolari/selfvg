import argparse

import pytorch_lightning as pl
import shortuuid


def get_args():
    parser = argparse.ArgumentParser()

    exp_group = parser.add_argument_group("experiment arguments")
    exp_group.add_argument(
        "--exp_id",
        type=str,
        default=shortuuid.uuid()[:4],
        help="Experiment identifier. Default: random",
    )
    exp_group.add_argument(
        "--mode",
        type=str,
        choices=["train"],
        default="train",
        help="Program mode. Default: train",
    )
    exp_group.add_argument(
        "--task",
        type=str,
        default="weak",
        choices=["weak", "full"],
        help="Model task, weakly- or fully-supervised. Default: weak",
    )
    exp_group.add_argument(
        "--max_epochs",
        type=int,
        default=25,
        help="Maximum number of epochs. Default: 25",
    )
    exp_group.add_argument(
        "--devices", type=int, default=1, help="Number of devices to use. Default: 1"
    )
    exp_group.add_argument(
        "--accelerator",
        type=str,
        choices=["cpu", "gpu"],
        default="gpu",
        help="Accelerator to use. Default: gpu",
    )
    exp_group.add_argument(
        "--logger",
        type=str,
        choices=["wandb", "tensorboard"],
        default="wandb",
        help="Logger to use for experiment tracking. Default: wandb",
    )
    exp_group.add_argument(
        "--dev",
        default=False,
        action="store_true",
        help="Dev mode uses validation set instead of training set. Default: false",
    )
    exp_group.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Verbose mode. Default: false",
    )

    # dataset params
    dataset_group = parser.add_argument_group("dataset arguments")
    dataset_group.add_argument(
        "--batch_size", type=int, default=32, help="Batch size. Default: 32"
    )
    dataset_group.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers. Default: 4"
    )
    dataset_group.add_argument(
        "--train_fraction",
        type=float,
        default=1.0,
        help="Fraction of training set to use. Default: 1.0",
    )

    # model params
    model_group = parser.add_argument_group("dataset arguments")
    model_group.add_argument(
        "--omega",
        type=float,
        default=0.5,
        help="Weight for the network prediction. Default: 0.5",
    )

    args = parser.parse_args()

    if args.dev:
        # force verbose logging in dev mode
        args.verbose = True

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
            project="weakvg",
            entity="weakly_guys",
            log_model=False if args.dev else True,
            name=name,
            settings=wandb.Settings(start_method="fork"),
        )

    return None


def get_callbacks(args):
    model_checkpoint_clbk = pl.callbacks.model_checkpoint.ModelCheckpoint(
        dirpath=f"output/{args.exp_id}",
        filename="{epoch}-{step}-{val_acc:.2f}",
        monitor="val_acc",
        mode="max",
        save_last=True,
    )

    callbacks = [
        model_checkpoint_clbk,
    ]

    return callbacks
