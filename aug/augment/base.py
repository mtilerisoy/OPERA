from abc import ABC, abstractmethod
import numpy as np

class BaseAugmentation(ABC):
    """
    Abstract base class for all audio augmentations.
    All augmentations must implement the __call__ method.
    """
    @abstractmethod
    def __call__(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply the augmentation to the input audio.
        
        Args:
            audio (np.ndarray): Input audio signal.
            **kwargs: Additional arguments for augmentation.
        
        Returns:
            np.ndarray: Augmented audio signal.
        """
        pass 