import multiprocessing
import logger as log
import model_types
import train
import argparse


def parse_arguments():
  """Parse command line arguments."""
  parser = argparse.ArgumentParser(
    description="Train a CNN model on the Animals10 dataset.")

  parser.add_argument(
    "--model_type",
    type=str,
    required=True,
    choices=["TINY_VGG", "RESNET18", "VGG16"],
    help="Choose the model architecture."
  )

  parser.add_argument(
    "--model_name",
    type=str,
    default="model",
    help="Filename to save the trained model and logs."
  )

  parser.add_argument(
    "--num_epochs",
    type=int,
    default=20,
    help="Number of training epochs."
  )

  parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size."
  )

  parser.add_argument(
    "--hidden_units",
    type=int,
    default=10,
    help="Hidden units."
  )

  parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.001,
    help="Learning rate."
  )

  parser.add_argument(
    "--train_dir",
    type=str,
    default="./input/train",
    help="Path to training data."
  )

  parser.add_argument(
    "--test_dir",
    type=str,
    default="./input/test",
    help="Path to test data."
  )

  return parser.parse_args()

def setup_logger(model_name: str):
  """Initialize and return a logger."""
  log_path = f"logs/{model_name}.log"
  return log.init_logger(log_path)

def main():
  # Argument parser setup
  args = parse_arguments()
  # Logger setup
  logger = setup_logger(args.model_name)
  logger.info("Training model started ..")

  multiprocessing.freeze_support()

  # MODEL SETUP
  MODEL_NAME = args.model_name
  MODEL_TYPE = getattr(model_types.ModelType, args.model_type)
  HIDDEN_UNITS = args.hidden_units
  NUM_EPOCHS = args.num_epochs
  BATCH_SIZE = args.batch_size
  LEARNING_RATE = args.learning_rate
  TRAIN_DIR = args.train_dir
  TEST_DIR = args.test_dir

  train.train_model(num_epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE,
                    hidden_units=HIDDEN_UNITS,
                    learning_rate=LEARNING_RATE,
                    model_name=MODEL_NAME,
                    model_type=MODEL_TYPE,
                    train_dir=TRAIN_DIR,
                    test_dir=TEST_DIR)

  logger.info("Training model finished ..")

if __name__ == '__main__':
  main()

