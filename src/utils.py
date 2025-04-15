import torch
from pathlib import Path
import logging

logger = logging.getLogger("logger")

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """
    Saves a trained PyTorch model to a specified directory.

    This function creates the target directory if it doesn't exist,
    checks that the filename has the correct extension, and then saves
    the model's state dictionary to the given path.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        target_dir (str): Directory path where the model will be saved.
        model_name (str): Name of the file to save the model as.
                          Must end with ".pt" or ".pth".

    Raises:
        AssertionError: If the provided model_name does not end with '.pt' or '.pth'.

    Example:
        save_model(model=my_model,
                   target_dir="models",
                   model_name="my_model.pth")
    """

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    logger.info(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)
