import torch
import torch.nn as nn
import torchvision.models as models
import logging
import model_types

logger = logging.getLogger("logger")


def create_model(model_type: model_types.ModelType, hidden_units: int, num_classes: int,
                 device):
  """
  Creates and returns a PyTorch model instance based on the specified model type.

  This function supports three model types:
  - TinyVGG: a custom CNN architecture.
  - ResNet18: a pretrained ResNet18 model from torchvision, modified for classification.
  - VGG16: a pretrained VGG16 model from torchvision, modified for classification.

  The model is moved to the specified device (CPU or GPU).

  Args:
      model_type (model_types.ModelType): The type of model to create (e.g., TINY_VGG, RESNET18, VGG16).
      hidden_units (int): Number of hidden units (only used for TinyVGG).
      num_classes (int): Number of output classes for the classification task.
      device (torch.device or str): The target device to move the model to (e.g., "cpu" or "cuda").

  Returns:
      torch.nn.Module: A PyTorch model ready for training or inference.

  Raises:
      ValueError: If an unsupported model type is passed.

  Example:
      model = create_model(model_type=ModelType.RESNET18,
                           hidden_units=128,
                           num_classes=10,
                           device="cuda")
  """

  if model_type == model_types.ModelType.TINY_VGG:
    model = TinyVGG(
      input_shape=3,
      hidden_units=hidden_units,
      output_shape=num_classes
    ).to(device)
  elif model_type == model_types.ModelType.RESNET18:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.to(device)
  elif model_type == model_types.ModelType.VGG16:
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
  else:
    logger.error(f"Unsupported model type: {model_type}")
    raise ValueError(f"Unsupported model type: {model_type}")

  return model

class TinyVGG(nn.Module):
  """
    Implements the TinyVGG convolutional neural network architecture in PyTorch.

    This model structure is inspired by the TinyVGG example on the CNN Explainer website:
    https://poloclub.github.io/cnn-explainer/

    Architecture Summary:
    - Two convolutional blocks, each containing two Conv2D layers with ReLU activation and a MaxPooling layer.
    - A fully connected (Linear) output layer after flattening the feature maps.

    Args:
        input_shape (int): Number of input channels (e.g., 3 for RGB images).
        hidden_units (int): Number of convolutional filters (feature maps) used in each Conv2D layer.
        output_shape (int): Number of output classes for classification.

    Example:
        model = TinyVGG(input_shape=3, hidden_units=64, output_shape=10)
  """

  def __init__(self, input_shape: int, hidden_units: int,
               output_shape: int) -> None:
    super().__init__()
    self.conv_block_1 = nn.Sequential(
      nn.Conv2d(in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0),
      nn.ReLU(),
      nn.Conv2d(in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2,
                   stride=2)
    )
    self.conv_block_2 = nn.Sequential(
      nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
      nn.ReLU(),
      nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
      nn.ReLU(),
      nn.MaxPool2d(2)
    )
    self.classifier = nn.Sequential(
      nn.Flatten(),
      # Where did this in_features shape come from?
      # It's because each layer of our network compresses and changes the shape of our inputs data.
      nn.Linear(in_features=hidden_units * 13 * 13,
                out_features=output_shape)
    )

  def forward(self, x: torch.Tensor):
    # operator fusion
    return self.classifier(self.block_2(self.block_1(x)))
