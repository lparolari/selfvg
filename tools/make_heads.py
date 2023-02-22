import json
import logging
import argparse

from weakvg.dataset import Flickr30kDataset
from weakvg.wordvec import get_nlp, get_objects_vocab, get_tokenizer, get_wordvec


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val")

    args = parser.parse_args()

    split = args.split

    tokenizer = get_tokenizer()
    _, vocab = get_wordvec(custom_tokens=get_objects_vocab())
    nlp = get_nlp()

    dataset = Flickr30kDataset(
        split=split,
        data_dir="data/flickr30k",
        tokenizer=tokenizer,
        vocab=vocab,
        nlp=nlp,
        transform=None,
    )

    heads = {}

    for datum in dataset.data:
        heads[datum.identifier] = [
            datum.get_heads(sentence_id) for sentence_id in datum.get_sentences_ids()
        ]

    json.dump(heads, open(f"data/flickr30k/{split}_heads.json", "w"))


if __name__ == "__main__":
    main()
