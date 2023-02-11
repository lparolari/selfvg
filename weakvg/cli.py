import argparse

import pytorch_lightning as pl
import shortuuid


def get_args():
    parser = argparse.ArgumentParser()

    # global params
    parser.add_argument("--exp_id", type=str, default=shortuuid.uuid()[:4])
    parser.add_argument("--mode", type=str, choices=["train"], default="train")
    parser.add_argument(
        "--dev",
        default=False,
        action="store_true",
        help="Dev mode uses validation set instead of training set",
    )
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument(
        "--logger", type=str, choices=["wandb", "tensorboard"], default="wandb"
    )
    parser.add_argument("--max_epochs", type=int, default=25)
    parser.add_argument(
        "--task",
        type=str,
        default="weak",
        choices=["weak", "full"],
        help="Task to perform: `weak` runs the model with weak supervision, while `full` uses supervision",
    )

    # dataset params
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_fraction", type=float, default=1.0)

    # model params
    parser.add_argument(
        "--omega", type=float, default=0.5, help="Weight for the network prediction"
    )

    args = parser.parse_args()

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
            log_model=True,
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
