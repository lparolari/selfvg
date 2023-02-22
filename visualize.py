import argparse
import logging

import cv2
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from weakvg.dataset import Flickr30kDataset, NormalizeCoord, collate_fn
from weakvg.model import MyModel
from weakvg.wordvec import get_nlp, get_objects_vocab, get_tokenizer, get_wordvec

vocab = None
font_size = 14
text_props = dict(facecolor="blue", alpha=0.5)

mpl.rcParams["figure.dpi"] = 100


def main():
    global vocab

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--search", type=str, default=None)

    args = parser.parse_args()

    checkpoint = args.checkpoint
    search = args.search
    split = args.split

    pl.seed_everything(42, workers=True)

    tokenizer = get_tokenizer()
    wordvec, vocab = get_wordvec(custom_tokens=get_objects_vocab())
    nlp = get_nlp()

    dataset = Flickr30kDataset(
        split=split,
        data_dir="data/flickr30k",
        tokenizer=tokenizer,
        vocab=vocab,
        nlp=nlp,
        transform=NormalizeCoord(),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    model = MyModel.load_from_checkpoint(checkpoint, wordvec=wordvec, vocab=vocab)

    logging.info("Model hparams:", model.hparams)

    for batch in dataloader:
        # forward the model
        scores, (multimodal_scores, concepts_scores) = model(batch)

        # compute candidates box for the full, multimodal and concepts model
        model_candidates, model_candidates_idx = model.predict_candidates(
            scores, batch["proposals"]
        )

        multimodal_candidates, multimodal_candidates_idx = model.predict_candidates(
            multimodal_scores, batch["proposals"]
        )

        concepts_candidates, concepts_candidates_idx = model.predict_candidates(
            concepts_scores, batch["proposals"]
        )

        # "unbatch" data -> remove batch dimension (getting the first element)
        # and convert to numpy arrays
        meta = unbatch(batch["meta"])
        targets = unbatch(batch["targets"])
        queries = unbatch(batch["queries"])
        image_w = unbatch(batch["image_w"])
        image_h = unbatch(batch["image_h"])
        sentence = unbatch(batch["sentence"])
        labels = unbatch(batch["labels"])

        identifier = meta[1]  # 0: idx, 1: identifier

        model_candidates, model_candidates_idx = unbatch(model_candidates), unbatch(
            model_candidates_idx
        )
        multimodal_candidates, multimodal_candidates_idx = unbatch(
            multimodal_candidates
        ), unbatch(multimodal_candidates_idx)
        concepts_candidates, concepts_candidates_idx = unbatch(
            concepts_candidates
        ), unbatch(concepts_candidates_idx)

        # get the labels for each candidate
        model_candidates_label = np.take(labels, model_candidates_idx, axis=0)
        multimodal_candidates_label = np.take(labels, multimodal_candidates_idx, axis=0)
        concepts_candidates_label = np.take(labels, concepts_candidates_idx, axis=0)

        # get the image to show on plot
        image = get_image(identifier)

        gt_color = np.array([255, 102, 102]) / 255  # #ff6666 (red)
        model_color = np.array([153, 255, 153]) / 255  # #99ff99 (green)
        concept_color = np.array([69, 205, 255]) / 255  # #45cdff (blue)
        mm_color = np.array([255, 255, 153]) / 255  # #ffff99 (yellow)

        sentence_txt = txt(sentence)
        model_candidates_label_txt = txt(model_candidates_label).split()
        multimodal_candidates_label_txt = txt(multimodal_candidates_label).split()
        concepts_candidates_label_txt = txt(concepts_candidates_label).split()

        if search is not None and search not in sentence_txt:
            continue

        # x1, y1, x2, y2
        img_size = [
            image_w,
            image_h,
            image_w,
            image_h,
        ]

        for i in range(len(queries)):
            query = queries[i]

            query_txt = txt(query)
            candidate_lab_txt = model_candidates_label_txt[i]
            concepts_lab_txt = concepts_candidates_label_txt[i]
            mm_lab_txt = multimodal_candidates_label_txt[i]

            # rescale boxes to image size
            target_box = targets[i] * img_size
            candidate_box = model_candidates[i] * img_size
            mm_box = multimodal_candidates[i] * img_size
            concepts_box = concepts_candidates[i] * img_size

            iou_score = iou(target_box, candidate_box)

            plt.subplot(1, len(queries), i + 1)
            plt.imshow(image)
            plt.title(
                f"{query_txt} ({round(iou_score, ndigits=2)})",
                fontdict={"fontsize": font_size},
                y=-0.1,
            )
            plt.suptitle(sentence_txt, fontsize=14)

            ax = plt.gca()

            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

            draw_box(ax, mm_box, mm_color, label=mm_lab_txt)
            draw_box(ax, concepts_box, concept_color, label=concepts_lab_txt)
            draw_box(ax, target_box, gt_color)
            draw_box(
                ax, candidate_box, model_color, label=candidate_lab_txt, dashed=True
            )

        plt.show()


def unbatch(x):
    return x[0].cpu().numpy()


def get_image(identifier):
    path = f"data/flickr30k/flickr30k_images/{identifier}.jpg"

    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    return im


def txt(tokens):
    return " ".join(vocab.lookup_tokens([t for t in tokens if t != 0]))


def draw_box(ax, box, color, label=None, dashed=False):
    x1, y1 = box[0], box[1]
    x2, y2 = box[2], box[3]
    xy = (x1, y1)
    width, height = x2 - x1, y2 - y1

    rect = patches.Rectangle(
        xy,
        width,
        height,
        linewidth=2,
        edgecolor=[*color, 1.0],
        facecolor=[*color, 0.2],
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(rect)

    if label is not None:
        ax.text(
            x1,
            y1,
            label,
            bbox={"facecolor": [*color, 1.0], "edgecolor": [*color, 0.0], "pad": 1.0},
            fontsize=font_size,
            color="black",
        )


def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union


if __name__ == "__main__":
    main()
