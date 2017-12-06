from typing import Iterable
from itertools import zip_longest

import numpy as np


POSITIVE = 0
NEGATIVE = 1


class ConfusionMatrix(np.ndarray):
    def __new__(cls, **kwargs):
        obj = np.zeros([2, 2], dtype=np.int32).view(cls)
        return obj

    def __init__(self):
        pass

    @property
    def true_positive(self):
        return self[POSITIVE][POSITIVE]

    @true_positive.setter
    def true_positive(self, value):
        self[POSITIVE][POSITIVE] = value

    @property
    def false_positive(self):
        return self[POSITIVE][NEGATIVE]

    @false_positive.setter
    def false_positive(self, value):
        self[POSITIVE][NEGATIVE] = value

    @property
    def false_negative(self):
        return self[NEGATIVE][POSITIVE]

    @false_negative.setter
    def false_negative(self, value):
        self[NEGATIVE][POSITIVE] = value

    @property
    def true_negative(self):
        return self[NEGATIVE][NEGATIVE]

    @true_negative.setter
    def true_negative(self, value):
        self[NEGATIVE][NEGATIVE] = value

    def precision(self):
        return self.true_positive / (self.true_positive + self.false_positive)

    def recall(self):
        return self.true_positive / (self.true_positive + self.false_negative)


def f1(precision, recall):
    return 2 * precision * recall / (precision + recall)


def confusion_matrix_from_sets(target: set, prediction: set) -> ConfusionMatrix:
    ret = ConfusionMatrix()
    ret.true_positive = len(target & prediction)
    ret.false_positive = len(prediction - target)
    ret.false_negative = len(target - prediction)
    ret.true_negative = 0
    return ret


class Example:
    def __init__(self, key: str, tags_string: str):
        self.key = key
        self.tags = {i for i in tags_string.split()}


def confusion_matrix_from_iterables(targets: Iterable[Example], predictions: Iterable[Example]) -> ConfusionMatrix:
    targets_buffer = {}
    predictions_buffer = {}
    cm = ConfusionMatrix()

    for target, prediction in zip_longest(targets, predictions):
        if target is None or prediction is None:
            targets_left = sum(1 for _ in targets)
            predictions_left = sum(1 for _ in predictions)
            diff = targets_left - predictions_left
            raise ValueError("targets and predictions should have the same length ({} more targets)".format(diff))
        assert isinstance(target, Example)
        assert isinstance(prediction, Example)

        if target.key == prediction.key:
            cm += confusion_matrix_from_sets(target.tags, prediction.tags)
        else:
            if target.key in predictions_buffer:
                cm += confusion_matrix_from_sets(target.tags, predictions_buffer.pop(target.key))
            else:
                targets_buffer[target.key] = target.tags

            if prediction.key in targets_buffer:
                cm += confusion_matrix_from_sets(targets_buffer.pop(prediction.key), prediction.tags)
            else:
                predictions_buffer[prediction.key] = prediction.tags

    if len(targets_buffer) == len(predictions_buffer) == 0:
        return cm
    else:
        raise ValueError("key of examples from targets and predictions did not match")
