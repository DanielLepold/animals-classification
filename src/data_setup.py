import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):
  """
  Creates PyTorch DataLoaders for training and testing datasets.

  This function takes in paths to directories containing training and testing images,
  applies the provided image transformations using torchvision, and wraps the datasets
  into DataLoaders for efficient batch processing during model training and evaluation.

  Args:
      train_dir (str): Path to the directory with training images.
                       Images should be organized in subfolders by class.
      test_dir (str): Path to the directory with testing images.
                      Same structure as `train_dir`.
      transform (transforms.Compose): A set of torchvision transforms to apply to both datasets.
      batch_size (int): Number of samples per batch.
      num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to NUM_WORKERS.

  Returns:
      Tuple[DataLoader, DataLoader, List[str]]:
          - train_dataloader: DataLoader for training data.
          - test_dataloader: DataLoader for test data.
          - class_names: A list of class names inferred from the folder structure.

  Example:
      train_loader, test_loader, classes = create_dataloaders(
          train_dir="data/train",
          test_dir="data/test",
          transform=my_transforms,
          batch_size=32,
          num_workers=4
      )
  """
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  class_names = train_data.classes

  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names
