# 🐾 Animal Image Classification with PyTorch

The goal of this project is to build a deep learning model that can classify images of animals into different categories.
The dataset containing images of 10 different animals (e.g., cat, dog, horse, etc.).

This project implements an image classification pipeline for animal images using PyTorch. It supports multiple model architectures (TinyVGG, ResNet18, VGG16) and includes training, evaluation, and model saving utilities.

## Project Structure

```aiignore
📁 src/
├── input/
│   ├── test/            # Input test data
│   └── train/           # Input train data
├── logs/
│   └── your_logs.pth    # Saved logs during running
├── models/
│   └── your_models.pth  # Saved trained models
├── data_setup.py        # Contains logic to prepare DataLoaders
├── engine.py            # Training and evaluation loop functions
├── helper.py            # Helper function for reorganisation of train,test data
├── logger.py            # Logging configuration
├── main.py              # Main
├── model_builder.py     # Functions to create and configure models
├── model_types.py       # Enum class defining supported model architectures
├── predictions.py       # Visualization
├── train.py             # Training script
└── utils.py             # Utility functions (e.g., saving models)
README.md                # You're here!
requirements.txt         # Packages to install
```

## 🐾 Dataset

This project uses the [Animals10 dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10) from Kaggle.
The dataset contains 10 categories of animal images such as cat, dog, horse, sheep, elephant, butterfly, chicken, cow, spider, and squirrel.

First you should save your kaggle authentication data (your json file downloaded from the site) on your computer:

```aiignore
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```
After that, you can download the dataset directly:

```aiignore
!kaggle datasets download -d alessiocorrado99/animals10 --unzip
```
### Organize the dataset into `train` and `test` folders:

After downloading, run the helper script to automatically split the dataset into training and testing sets:

```bash
python helper.py
```

After running the script, your folder structure should look like this:

```
./input/train/<class_names>/
./input/test/<class_names>/
```

## Supported Architectures

- 🟦 `TinyVGG`: A simple CNN architecture modeled after the [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
- 🟥 `ResNet18`: Pretrained model from `torchvision.models`
- 🟩 `VGG16`: Pretrained model from `torchvision.models`

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

## 🚀 Running the Training Script

To train a model, run the following script from the root directory, where the main is located:

```bash
python main.py --model_type MODEL_TYPE
```

`MODEL_TYPE` is **required** and must be one of the following:
- `TINY_VGG`
- `RESNET18`
- `VGG16`

### Optional Arguments

You can customize training with the following optional arguments:

| Argument          | Description                      | Default Value  |
|-------------------|----------------------------------|----------------|
| `--model_name`    | Name of the model and log file   | `model`        |
| `--num_epochs`    | Number of epochs to train        | `20`           |
| `--batch_size`    | Batch size used during training  | `32`           |
| `--hidden_units`  | The size of the hidden units     | `10`           |
| `--learning_rate` | Learning rate for optimizer      | `0.001`        |
| `--train_dir`     | Directory path for training data | `./input/train` |
| `--test_dir`      | Directory path for test data     | `./input/test` |

### 🧪 Example

```bash
python main.py --model_type TINY_VGG --model_name TINY_VGG4 --num_epochs 20 --hidden_units 32 --train_dir "./data/train" --test_dir "./data/test"
```

This command will train the `TINY_VGG` model for 20 epochs with a learning rate of `0.001` and save the model as `TINY_VGG4.pth`.

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
