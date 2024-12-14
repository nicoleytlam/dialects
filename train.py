"""
Code for Problems 15 and 16.
"""
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import csv

from _utils import timer
from data_loader import Dataset, Vocabulary
from metrics import AverageLoss, Accuracy
from encoder_decoder import Seq2Seq


def print_metrics(loss: float, accuracy: float, message: str):
    print("{}. Loss: {:.2f}, Accuracy: {:.3f}".format(message, loss, accuracy))


def train_epoch(model: Seq2Seq, train_data: Dataset, vocab: Vocabulary,
                batch_size: int, loss_function: nn.CrossEntropyLoss, 
                optimizer: optim.Adam, pad_left: bool=False, teacher_forcing_ratio: float = .5,
                report_frequency: int = 1000):
    """
    Problem 16: Please complete this function, which trains a Seq2Seq model
    for one epoch. 

    :param model: The model that will be trained
    :param train_data: The training data
    :param vocab: a Vocabulary object
    :param batch_size: The batch size
    :param loss_function: The loss function
    :param optimizer: The Adam optimizer
    :param pad_left: pad extra positions at left of string
    :param teacher_forcing_ratio: proportion of teacher forcing
    """
    model.train()  # This needs to be called before the model is trained

    pad_index = vocab.get_index('[PAD]')
    loss_metric = AverageLoss()
    acc_metric = Accuracy(pad_index)

    batches = train_data.get_batches(batch_size, pad_input_left=pad_left)
    for i, (input, target) in enumerate(batches):
        # Problem 16: Replace the following two lines with your code.
        # The variable batch_loss must contain the loss incurred for the
        # current mini-batch. The variable output should contain the
        # output of model. Do not edit anything in this function above
        # this line.
        optimizer.zero_grad()
        output, _ = model(input, target, teacher_forcing_ratio=teacher_forcing_ratio)
    
        # Initialize batch loss
        batch_loss = 0.0
        for pos in range(target.shape[1]):  # Loop over sequence length
            batch_loss += loss_function(output[:, pos, :], target[:, pos])
        
        # Backpropagation
        batch_loss.backward()
        optimizer.step()
            
        
        # Update metrics. Do not edit anything in this function below
        # this line.
        avg_batch_loss = loss_metric.update(batch_loss, 
                                            target.shape[0] * target.shape[1] - 
                                            torch.bincount(target.view((-1,)))[pad_index])
        batch_acc = 0.
        for pos in range(output.shape[1]):
            batch_acc += acc_metric.update(output[:,pos,:],target[:,pos])
        batch_acc = batch_acc / output.shape[1]
        if (i + 1) % report_frequency == 0:
            print_metrics(avg_batch_loss, batch_acc, "Batch {}".format(i + 1))
            # print([vocab.get_form(w) for w in input[0]])
            # print([vocab.get_form(w) for w in target[0]])
            # print([vocab.get_form(w) for w in output.argmax(dim=-1)[0]])

    # Report epoch results
    print_metrics(loss_metric.value, acc_metric.value, "Training Complete")


def evaluate(model: Seq2Seq, test_data: Dataset,
             loss_function: nn.CrossEntropyLoss, message: str,
             pad_index: int = 1, pad_left: bool = False) \
        -> Tuple[float, float]:
    """
    Problem 17: Please complete this function, which evaluates a model
    using a training or development set.

    :param model: The model that will be evaluated
    :param test_data: The data to evaluate the model on
    :param loss_function: The loss function

    DO NOT USE THESE PARAMETERS:

    :param message: A message to be displayed when showing results
    :param pad_index: POS tag index to be ignored in evaluation
    :param pad_left: whether to pad batches on the left

    :return: The average loss and accuracy attained by model on
        test_data
    """
    model.eval()  # This needs to be called before the model is evaluated

    loss_metric = AverageLoss()
    acc_metric = Accuracy(pad_index)
    pred_list: List[int] = []
    target_list: List[int] = []

    for i, (input, target) in enumerate(test_data.get_batches(100,pad_input_left=pad_left)):
        # Problem 17: Replace the following two lines with your code.
        # The variable batch_loss must contain the loss incurred for the
        # current mini-batch. The variable output must contain the
        # output of model. Do not edit anything in this function
        # above this line.
        output, _ = model(input, target, teacher_forcing_ratio=0)

        batch_loss = 0.0
        for pos in range(target.shape[1]):
            batch_loss += loss_function(output[:, pos, :], target[:, pos])

        # Update metrics. Do not edit anything in this function below
        # this line.
        _ = loss_metric.update(batch_loss, target.shape[0] * target.shape[1] - 
                                            torch.bincount(target.view((-1,)))[pad_index])
        for pos in range(output.shape[1]):
            _ = acc_metric.update(output[:,pos,:],target[:,pos])
   
    # Report epoch results
    print_metrics(loss_metric.value, acc_metric.value, message)
    return loss_metric.value, acc_metric.value


def run_trial(model: Seq2Seq, train_data: Dataset,
              dev_data: Dataset, test_data: Dataset, vocab: Vocabulary, 
              filename: str = "checkpoint.pt", num_epochs: int = 10,
              batch_size: int = 5, lr: float = .001, pad_left: bool = False, 
              teacher_forcing_ratio: float = .5, report_frequency: int = 1000,
              dataname: str = "standard", config: str = "B") \
        -> Seq2Seq:
    """
    Problems 17–21: Use this function to train a model with one
    particular configuration of hyperparameters.

    DATASET HYPERPARAMETERS:

    :param train_data: The training data
    :param dev_data: The validation data
    :param test_data: The testing data
    :param pad_index:

    MODEL HYPERPARAMETERS:

    :param model: The model that will be trained
    :param filename: The filename to save checkpoints to. It should end
        with ".pt"

    TRAINING LOOP HYPERPARAMETERS:

    :param num_epochs: The number of epochs to train the model for
    :param batch_size: The size of the mini-batches used to train the
        model

    ADAM HYPERPARAMETERS:

    :param lr: The initial learning rate used for Adam

    :return: The trained model
    """
    # Create model
    pad_index = vocab.get_index('[PAD]')
    loss_function = nn.CrossEntropyLoss(ignore_index=pad_index, reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    best_dev_acc = 0.
    best_epoch = 0.
    
    row = [dataname, config]

    for i in range(1, num_epochs+1):
        print("Epoch {}".format(i))
        model.epoch = i
        # Train model
        train_epoch(model, train_data, vocab, batch_size, loss_function, optimizer, 
                    pad_left=pad_left, teacher_forcing_ratio=teacher_forcing_ratio, report_frequency=report_frequency)
        _, dev_acc = evaluate(model, dev_data, loss_function,
                              "Epoch {} Validation".format(i),
                              pad_index, pad_left=pad_left)

        
        # Save checkpoint
        row.append(dev_acc)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_epoch = i
            
            with timer("Saving checkpoint..."):
                torch.save(model.state_dict(), filename)

    print("The best validation accuracy of {:.3f} occurred after epoch {}."
          "".format(best_dev_acc, best_epoch))
    row.append(best_dev_acc)

    # model.load_state_dict(torch.load(filename, weights_only=True))
    model.load_state_dict(torch.load(filename))
    _, test_acc = evaluate(model, test_data, loss_function, "Test", pad_index, pad_left=pad_left)
    row.append(test_acc)


    # Open the file in append mode
    with open('accuracy.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
    
    print(row)
    # return model
