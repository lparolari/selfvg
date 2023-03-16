from typing import List

Box = List[int]


class UpperboundAccuracyMixin:
    def get_upperbound_accuracy(self):
        total = 0
        matched = 0

        for sample in iter(self):
            targets = sample["targets"]
            proposals = sample["proposals"]

            for target in targets:
                iou_scores = [iou(target, proposal) for proposal in proposals]
                max_iou = max(iou_scores)

                if max_iou >= 0.5:
                    matched += 1

                total += 1

        return matched / total


class QueryNumberMixin:
    def get_min_query_number(self):
        return min(len(sample["queries"]) for sample in iter(self))
    
    def get_max_query_number(self):
        return max(len(sample["queries"]) for sample in iter(self))

    def get_avg_query_number(self):
        return sum(len(sample["queries"]) for sample in iter(self)) / len(self)


def iou(box_a: Box, box_b: Box):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union
