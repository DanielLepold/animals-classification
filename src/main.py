import multiprocessing
import torch
import data_setup, engine, model_builder, utils
import logger as log
import model_types
from torchvision import transforms


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

  # Create transforms
  data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
  ])

  # Create DataLoaders with help from data_setup.py
  train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
  )

  # Create model with help from model_builder.py
  model = model_builder.create_model(model_type=model_type,
                                     hidden_units=HIDDEN_UNITS,
                                     num_classes=len(class_names),
                                     device=device)
  logger.info(f"Number of classes = {len(class_names)}")
  logger.info(f"Model type: {model_type}")
  logger.info(f"Model configuration: \n{model}")

  # Set loss and optimizer
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=LEARNING_RATE)

  # Start training with help from engine.py
  engine.train(model=model,
               train_dataloader=train_dataloader,
               test_dataloader=test_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               epochs=NUM_EPOCHS,
               device=device)

  # Save the model with help from utils.py
  utils.save_model(model=model,
                   target_dir="models",
                   model_name=f"{model_name}.pth")
  pass

if __name__ == '__main__':
  # MODEL SETUP
  MODEL_NAME = "VGG16_model"
  MODEL_TYPE = model_types.ModelType.VGG16
  log_path = f"logs/{MODEL_NAME}.log"
  logger = log.init_logger(log_path)
  logger.info("Training model started .. ")
  multiprocessing.freeze_support()

  # Setup directories
  TRAIN_DIR = "./input/train"
  TEST_DIR = "./input/test"

  # Setup hyperparameters
  NUM_EPOCHS = 20
  BATCH_SIZE = 32
  LEARNING_RATE = 0.001

  # In case of TINY_VGG
  if MODEL_TYPE == model_types.ModelType.TINY_VGG:
    HIDDEN_UNITS = 30
  else:
    HIDDEN_UNITS = None

  train_model(num_epochs=NUM_EPOCHS,
              batch_size=BATCH_SIZE,
              hidden_units=HIDDEN_UNITS,
              learning_rate=LEARNING_RATE,
              model_name=MODEL_NAME,
              model_type=MODEL_TYPE,
              train_dir=TRAIN_DIR,
              test_dir=TEST_DIR)

  logger.info("Training model finished .. ")
