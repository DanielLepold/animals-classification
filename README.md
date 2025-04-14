# 🐾 Animal Image Classification with PyTorch

The goal of this project is to build a deep learning model that can classify images of animals into different categories.
The dataset containing images of 10 different animals (e.g., cat, dog, horse, etc.).

This project implements an image classification pipeline for animal images using PyTorch. It supports multiple model architectures (TinyVGG, ResNet18, VGG16) and includes training, evaluation, and model saving utilities.

## Project Structure

```aiignore
📁 src/
├── data_setup.py        # Contains logic to prepare DataLoaders
├── engine.py            # Training and evaluation loop functions
├── model_builder.py     # Functions to create and configure models
├── input/
│   ├── test/            # Input test data
│   └── train/           # Input train data
├── models/
│   └── your_model.pth   # Saved trained models
├── logs/
│   └── your_logs.pth    # Saved logs during running
├── main.py              # Main training script
├── predictions.py       # Visualization
├── utils.py             # Utility functions (e.g., saving models)
├── logger.py            # Logging configuration
└── model_types.py       # Enum class defining supported model architectures
README.md                # You're here!
```

## 🐾 Dataset

This project uses the [Animals10 dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10) from Kaggle.
The dataset contains 10 categories of animal images such as cat, dog, horse, sheep, elephant, butterfly, chicken, cow, spider, and squirrel.

You can download the dataset directly using:

```aiignore
kaggle datasets download -d alessiocorrado99/animals10 --unzip
```

## Supported Architectures

- 🟦 `TinyVGG`: A simple CNN architecture modeled after the [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
- 🟥 `ResNet18`: Pretrained model from `torchvision.models`
- 🟩 `VGG16`: Pretrained model from `torchvision.models`

```aiignore

Each subfolder inside `train/` and `test/` represents a class label.

## Requirements

### Installation Instructions
To set up the project and install the necessary dependencies, follow these steps:

Create a virtual environment (optional but recommended):
```aiignore
python -m venv venv
```
Activate the virtual environment:
- For Windows:
```aiignore
.\venv\Scripts\activate
```
- For macOS/Linux:
```aiignore
source venv/bin/activate
```
**Install the dependencies:**
With your virtual environment activated, run the following command to install the dependencies from requirements.txt:
```aiignore
pip install -r requirements.txt
```
After installation, you can verify that the required packages are installed by running:
```aiignore
pip list
```

## Training

To train a model, use the `train_model()` function from `train.py`. Here's a sample usage:

```aiignore
train_model(
    num_epochs=10,
    batch_size=32,
    hidden_units=128,
    learning_rate=0.001,
    model_name="animal_classifier_tinyvgg",
    model_type=ModelType.TINY_VGG,
    train_dir="path/to/train",
    test_dir="path/to/test"
)
```

## Logs

All training logs are stored using a file logger created with init_logger() in logger.py.

## Evaluation

Accuracy and loss are printed during training, and evaluation is performed after each epoch.

## Saving models

Trained models are automatically saved to the models/ directory with the name format: <model_name>.pth.



## Results

You can track training loss and accuracy in the logs or by extending the code with visualization tools like Matplotlib or TensorBoard.



## 📈 Future improvements

Add support for early stopping and learning rate scheduling
Add training visualization with TensorBoard or Matplotlib
Export to ONNX or TorchScript ""
