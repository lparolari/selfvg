import logging
from typing import List, Tuple, Optional

import torch
from torchtext.vocab import GloVe, vocab as make_vocab

Alternative = Optional[List[str]]
Missing = Tuple[str, Alternative]


def fix_oov(labels: List[str], wordvec, vocab) -> List[Missing]:
    from spellchecker import SpellChecker

    missing = []  # list of pairs (label, alternative)

    spell = SpellChecker()

    unk_token = vocab.lookup_token(vocab.get_default_index())

    for label in labels:
        found_alternative = False
        alternative = []

        if label in vocab:
            continue

        # try to find an alternative embedding for the given label
        # a label may be something like "stop sign,stopsign"
        #
        # 1. split label by comma, we may find one of the parts in the vocab
        # 2. split each part by space, we may find one of the words in the vocab
        #    in this case is necessary to average over all words because we
        #    don't know which word is the most important
        # 3. try to correct spelling of each word if no previous match was found

        parts = label.split(",")

        for part in parts:
            if part in vocab:
                emb = wordvec[part]
                found_alternative = True
                alternative.append(part)

                break  # stop looking for alternative in other parts
            else:
                # part may be "stop sign", so we need to average over words

                words_found = 0.0
                emb = wordvec[unk_token]

                for word in part.split():
                    if word in vocab:
                        emb += wordvec[word]
                        words_found += 1.0
                        found_alternative = True
                        alternative.append(word)
                    else:
                        corrected = spell.correction(word)
                        if corrected in vocab:
                            emb += wordvec[corrected]
                            words_found += 1.0
                            found_alternative = True
                            alternative.append(corrected)

                if found_alternative:
                    emb = emb / words_found  # words_found should be > 0

                    break  # stop looking for alternative in other parts

        if found_alternative:
            logging.debug(
                f"Found alternative embedding for '{label}' using {len(alternative)} words: {', '.join(alternative)}"
            )

            missing.append((label, alternative))

            vocab.append_token(label)
            wordvec.vectors = torch.cat([wordvec.vectors, emb.unsqueeze(0)], dim=0)

        else:
            logging.debug(f"Alternative not found for '{label}'")
            missing.append((label, None))

    return missing


def get_wordvec(
    name="6B",
    dim=300,
    *,
    cache="data/glove",
    custom_tokens: List[str] = [],
    return_vocab=True,
):
    wordvec = GloVe(name, dim, cache=cache)
    vocab = build_vocab(wordvec)

    # add the embedding for the special token position 0
    padding_emb = torch.zeros(1, wordvec.dim)
    wordvec.vectors = torch.cat([padding_emb, wordvec.vectors], dim=0)

    missing = fix_oov(custom_tokens, wordvec, vocab)
    fixed = [label for label, alternative in missing if alternative is not None]

    if len(fixed) > 0:
        logging.info(f"Found {len(missing)} oov labels, fixed {len(fixed)}")

    if return_vocab:
        return wordvec, vocab

    return wordvec


def build_vocab(wordvec):
    """
    Build a vocab using GloVe tokens
    """
    unk_index = 0
    unk_token = "<unk>"

    # min_freq=0 is required to include the first token in `wordvec.stoi` which has index 0
    # the `make_vocab` function requires an ordered dict as input and discards entries
    # whose value is less than `min_freq`
    vocab = make_vocab(wordvec.stoi, specials=[unk_token], min_freq=0)
    
    vocab.set_default_index(unk_index)

    return vocab


def get_objects_vocab(path="data/objects_vocab.txt"):
    with open(path, "r") as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]
    return labels


def get_nlp():
    import spacy

    nlp = spacy.load("en_core_web_sm")
    return nlp


def get_tokenizer():
    from torchtext.data.utils import get_tokenizer as get_t

    return get_t("basic_english")
