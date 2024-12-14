"""
Code for computing loss and accuracy over multiple batches. DO NOT EDIT
THIS FILE.
"""
from abc import ABC, abstractmethod

import torch


class Metric(ABC):
    """
    An object that keeps track of performance metrics during training
    and testing.
    """

    @abstractmethod
    def reset(self):
        """
        Clears any data saved by this object.
        """
        pass

    @property
    @abstractmethod
    def value(self) -> float:
        """
        Retrieves the current value of the metric.

        :return: The current value of the metric
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> float:
        """
        Updates the metric after each mini-batch.

        :return: The value of the metric for the current mini-batch
        """
        pass


class AverageLoss(Metric):
    """
    Keeps track of the average loss incurred by the model per example.
    """

    def __init__(self):
        self.total_loss = 0.
        self.total_predictions = 0

    def reset(self):
        self.total_loss = 0.
        self.total_predictions = 0

    @property
    def value(self) -> float:
        if self.total_predictions == 0:
            return 0.
        else:
            return self.total_loss / self.total_predictions

    def update(self, loss: float, num_predictions: int) -> float:
        """
        Updates the average loss.

        :param loss: The total loss incurred by a mini-batch
        :param num_predictions: The number of predictions in the mini-
            batch (i.e., the batch size)
        :return: The average loss for the mini-batch
        """
        self.total_loss += loss
        self.total_predictions += num_predictions
        return loss / num_predictions


class Accuracy(Metric):
    """
    Keeps track of the average accuracy of the model.
    """

    def __init__(self, pad_index: int):
        self.num_correct = 0
        self.total_predictions = 0
        self.pad_index = pad_index

    def reset(self):
        self.num_correct = 0
        self.total_predictions = 0

    @property
    def value(self) -> float:
        if self.total_predictions == 0:
            return 0.
        else:
            return self.num_correct / self.total_predictions

    def update(self, logits: torch.FloatTensor, targets: torch.LongTensor) \
            -> float:
        """
        Updates the accuracy.

        :param logits: The output of a model. Shape: (batch size, vocab size)
        :param targets: The correct labels for the mini-batch. Shape:
            (batch size,)
        :return: The accuracy of the current batch
        """
        batch_num_correct = int(((logits.argmax(axis=-1) == targets) &
                                 (targets != self.pad_index)).sum())
        batch_num_predictions = targets.numel() - \
                                int((targets == self.pad_index).sum())
        batch_accuracy = batch_num_correct / batch_num_predictions

        self.num_correct += batch_num_correct
        self.total_predictions += batch_num_predictions

        return batch_accuracy
