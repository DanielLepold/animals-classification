import torch
import data_setup, engine, model_builder, utils
import model_types
from torchvision import transforms
import logging as log

logger = log.getLogger("logger")

def train_model(num_epochs: int,
                batch_size: int,
                hidden_units: int,
                learning_rate: float,
                model_name: str,
                model_type: model_types.ModelType,
                train_dir: str,
                test_dir: str):
  """
        Trains an image classification model using the specified configuration.

        Args:
            num_epochs (int): Number of training epochs.
            batch_size (int): Number of samples per training batch.
            hidden_units (int): Number of hidden units in the model (used for custom architectures).
            learning_rate (float): Learning rate for the optimizer.
            model_name (str): Name to use when saving the trained model (without file extension).
            model_type (model_types.ModelType): Type of model to create (e.g., custom CNN, VGG16, etc.).
            train_dir (str): Directory path containing training images organized in class-specific folders.
            test_dir (str): Directory path containing testing images organized in class-specific folders.

        Logs:
            Logs configuration details, model summary, and training progress.

        Saves:
            Trained model file in the 'models' directory with the name `{model_name}.pth`.

  """

  logger.info("Training configuration:")
  logger.info(f"  num_epochs     = {num_epochs}")
  logger.info(f"  batch_size     = {batch_size}")
  logger.info(f"  hidden_units   = {hidden_units}")
  logger.info(f"  learning_rate  = {learning_rate}")


  # Setup target device
  device = "cuda" if torch.cuda.is_available() else "cpu"


  if model_type == model_types.ModelType.VGG16:
    data_transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
  elif model_type == model_types.ModelType.RESNET18:
    data_transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
  else:
    data_transform = transforms.Compose([
      transforms.Resize((64, 64)),
      transforms.ToTensor()
    ])

  # Create DataLoaders with help from data_setup.py
  train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=batch_size
  )

  # Create model with help from model_builder.py
  model = model_builder.create_model(model_type=model_type,
                                     hidden_units=hidden_units,
                                     num_classes=len(class_names),
                                     device=device)

  logger.info(f"Number of classes = {len(class_names)}")
  logger.info(f"Model type: {model_type}")
  logger.info(f"Model configuration: \n{model}")

  # Set loss and optimizer
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=learning_rate)

  # Start training with help from engine.py
  engine.train(model=model,
               train_dataloader=train_dataloader,
               test_dataloader=test_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               epochs=num_epochs,
               device=device)

  # Save the model with help from utils.py
  utils.save_model(model=model,
                   target_dir="models",
                   model_name=f"{model_name}.pth")
  pass
