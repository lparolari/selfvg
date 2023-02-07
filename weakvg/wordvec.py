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
    check_oov: List[str] = [],
    return_vocab=True,
):
    wordvec = GloVe(name, dim, cache=cache)

    unk_index = 0
    unk_token = "<unk>"

    vocab = make_vocab(wordvec.stoi, specials=[unk_token])

    vocab.set_default_index(unk_index)

    missing = fix_oov(check_oov, wordvec, vocab)
    fixed = [label for label, alternative in missing if alternative is not None]

    logging.info(f"Found {len(fixed)} oov labels, fixed {len(fixed)}")

    if return_vocab:
        return wordvec, vocab

    return wordvec
