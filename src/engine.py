import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger("logger")

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
  """
  Trains a PyTorch model for one epoch.

  This function puts the model in training mode and performs a full pass over
  the provided DataLoader. For each batch, it runs the forward pass,
  computes the loss, performs backpropagation, and updates the model weights.

  Args:
    model (torch.nn.Module): The PyTorch model to train.
    dataloader (torch.utils.data.DataLoader): DataLoader providing the training data.
    loss_fn (torch.nn.Module): Loss function used to compute the difference between predictions and targets.
    optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
    device (torch.device): The device on which computations are performed (e.g., "cuda" or "cpu").

  Returns:
    Tuple[float, float]: A tuple containing the average training loss and training accuracy
    for the entire epoch.

  Example usage:
    train_loss, train_acc = train_step(
        model=model,
        dataloader=train_dataloader,
        loss_fn=loss_function,
        optimizer=optimizer,
        device=device
    )
  """
  # Put model in train mode
  model.train()

  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0

  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader):
    # Send data to target device
    X, y = X.to(device), y.to(device)

    # 1. Forward pass
    y_pred = model(X)

    # 2. Calculate  and accumulate loss
    loss = loss_fn(y_pred, y)
    train_loss += loss.item()

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    # Calculate and accumulate accuracy metric across all batches
    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    train_acc += (y_pred_class == y).sum().item() / len(y_pred)

  # Adjust metrics to get average loss and accuracy per batch
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
  """
  Evaluates a PyTorch model on a testing dataset for a single epoch.

  This function switches the model to evaluation mode (`model.eval()`), disables
  gradient computation to speed up inference and reduce memory usage, and performs
  a forward pass through the model on the test dataset. It calculates and returns
  the average loss and accuracy over the entire test set.

  Args:
      model (torch.nn.Module): The PyTorch model to be evaluated.
      dataloader (torch.utils.data.DataLoader): DataLoader containing the test data.
      loss_fn (torch.nn.Module): Loss function to compute the error on test predictions.
      device (torch.device): Device to perform computations on ("cuda" or "cpu").

  Returns:
      Tuple[float, float]: A tuple containing the average test loss and accuracy.
          Format: (test_loss, test_accuracy)

  Example:
      test_loss, test_acc = test_step(model, test_loader, loss_fn, device)
  """

  # Put model in eval mode
  model.eval()

  # Setup test loss and test accuracy values
  test_loss, test_acc = 0, 0

  # Turn on inference context manager
  with torch.inference_mode():
    # Loop through DataLoader batches
    for batch, (X, y) in enumerate(dataloader):
      # Send data to target device
      X, y = X.to(device), y.to(device)

      # 1. Forward pass
      test_pred_logits = model(X)

      # 2. Calculate and accumulate loss
      loss = loss_fn(test_pred_logits, y)
      test_loss += loss.item()

      # Calculate and accumulate accuracy
      test_pred_labels = test_pred_logits.argmax(dim=1)
      test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

  # Adjust metrics to get average loss and accuracy per batch
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
  """
  Trains and evaluates a PyTorch model over a number of epochs.

  This function loops through the training and testing phases for a specified
  number of epochs. For each epoch, it trains the model on the training dataset
  and evaluates it on the testing dataset. Metrics such as loss and accuracy
  are logged and stored throughout the process.

  Additionally, early stopping is implemented: if the training loss drops below
  0.2, the training stops early.

  Args:
      model (torch.nn.Module): The PyTorch model to be trained and tested.
      train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
      test_dataloader (torch.utils.data.DataLoader): DataLoader for testing data.
      optimizer (torch.optim.Optimizer): Optimizer to update model parameters.
      loss_fn (torch.nn.Module): Loss function to compute the error.
      epochs (int): Number of training epochs.
      device (torch.device): Device on which to perform computations ("cuda" or "cpu").

  Returns:
      Dict[str, List]: A dictionary containing training and testing metrics:
          {
              "train_loss": [<float>, <float>, ...],
              "train_acc": [<float>, <float>, ...],
              "test_loss": [<float>, <float>, ...],
              "test_acc": [<float>, <float>, ...]
          }

  Example:
      results = train(model, train_loader, test_loader, optimizer,
                      loss_fn, epochs=10, device=torch.device("cuda"))
  """

  # Create empty results dictionary
  results = {"train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": []
             }

  # Make sure model on target device
  model.to(device)

  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model=model,
                                       dataloader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       device=device)
    test_loss, test_acc = test_step(model=model,
                                    dataloader=test_dataloader,
                                    loss_fn=loss_fn,
                                    device=device)

    # Print out what's happening
    accuracy = (
      f"Epoch: {epoch + 1} | "
      f"train_loss: {train_loss:.4f} | "
      f"train_acc: {train_acc:.4f} | "
      f"test_loss: {test_loss:.4f} | "
      f"test_acc: {test_acc:.4f}"
    )

    print(accuracy)
    logger.info(accuracy)

    # Update results dictionary
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

    # Early stopping condition
    if train_loss < 0.1:
      logger.info(
        f"Stopping early at epoch {epoch + 1} because train_loss < 0.1")
      print(f"Stopping early at epoch {epoch + 1} because train_loss < 0.1")
      break  # Exit the training loop

  # Return the filled results at the end of the epochs
  return results
