"""Model Training Functions."""

import numpy as np
from torch import tensor
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train(
    data: DataLoader, model: Module, optimizer: Optimizer, loss_func: CrossEntropyLoss
) -> float:
    """Model Training Loop."""
    batch_loss = 0.0
    model.train()
    for _, batch in enumerate(data):
        images, labels = batch

        output = model(images)
        labels = np.array(labels)
        labels = labels.reshape(len(images), output.shape[1])
        labels = tensor(labels)
        loss = loss_func(output, labels)

        loss_val = loss.detach().item()
        batch_loss += loss_val

        loss.backward()
        optimizer.step()

    return batch_loss


def evaluate(data: DataLoader, model: Module, loss_func: CrossEntropyLoss) -> float:
    """Model Testing Loop."""
    model.eval()
    ev_epoch_loss = 0
    for _, batch in enumerate(data):
        inputs, labels = batch
        output = model(inputs)
        labels = np.array(labels)
        labels = labels.reshape(len(inputs), output.shape[1])
        labels = tensor(labels)

        loss = loss_func(output, labels)
        loss_val = loss.detach().item()
        ev_epoch_loss += loss_val
    return ev_epoch_loss
