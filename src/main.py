import multiprocessing
import logger as log
import model_types
import train


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

  train.train_model(num_epochs=NUM_EPOCHS,
              batch_size=BATCH_SIZE,
              hidden_units=HIDDEN_UNITS,
              learning_rate=LEARNING_RATE,
              model_name=MODEL_NAME,
              model_type=MODEL_TYPE,
              train_dir=TRAIN_DIR,
              test_dir=TEST_DIR)

  logger.info("Training model finished .. ")
