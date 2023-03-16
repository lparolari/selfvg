import argparse
import json
import logging
import os
import sys

# allow the script to be run from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weakvg.dataset import Flickr30kDataset, ReferitDataset
from weakvg.wordvec import get_nlp, get_objects_vocab, get_tokenizer, get_wordvec


def main():

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="flickr30k", choices=["flickr30k", "referit"]
    )
    parser.add_argument("--split", type=str, default="val")

    args = parser.parse_args()

    split = args.split
    dataset_name = args.dataset

    tokenizer = get_tokenizer()
    _, vocab = get_wordvec(custom_labels=get_objects_vocab())
    nlp = get_nlp()

    if dataset_name == "flickr30k":
        data_dir = "data/flickr30k"
        dataset = Flickr30kDataset(
            split=split,
            data_dir=data_dir,
            tokenizer=tokenizer,
            vocab=vocab,
            nlp=nlp,
            transform=None,
        )

    if dataset_name == "referit":
        data_dir = "data/referit"
        dataset = ReferitDataset(
            split=split,
            data_dir=data_dir,
            tokenizer=tokenizer,
            vocab=vocab,
            nlp=nlp,
            transform=None,
        )

    heads = {}

    for datum in dataset.data:
        heads[datum.identifier] = datum.get_heads()

    json.dump(heads, open(f"{data_dir}/{split}_heads.json", "w"))


if __name__ == "__main__":
    main()
