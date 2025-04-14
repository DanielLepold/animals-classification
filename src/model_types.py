from enum import Enum, auto

class ModelType(Enum):
    """
    Enumeration of supported model architectures.

    This enum is used to specify which model type should be created or trained.
    It helps ensure consistency and avoids the use of hard-coded strings throughout
    the codebase.

    Attributes:
        TINY_VGG: A custom TinyVGG model architecture.
        RESNET18: Pretrained ResNet-18 architecture from torchvision.
        VGG16: Pretrained VGG-16 architecture from torchvision.

    Example usage:
        model_type = ModelType.RESNET18
    """
    TINY_VGG = auto()
    RESNET18 = auto()
    VGG16 = auto()
