import random
from typing import List, Tuple
import numpy as np
from aug.augment.noise import NoiseAugmentation
from aug.augment.artefact import ArtefactAugmentation

class AugmentationPipeline:
    """
    Composes and applies augmentations to audio based on configuration and random selection.
    """
    def __init__(self, config, noise_files: List[str], artefact_specs: List[dict], sr: int, duration: float, noise_loader, noise_type: str):
        """
        Args:
            config: Configuration object or dictionary.
            noise_files (List[str]): List of available noise file names.
            artefact_specs (List[dict]): List of artefact specifications.
            sr (int): Sample rate.
            duration (float): Target duration in seconds.
            noise_loader (callable): Function to load noise audio by filename.
        """

        self.config = config
        
        self.noise_files = noise_files
        self.artefact_specs = artefact_specs
        
        self.sr = sr
        self.duration = duration
        self.noise_loader = noise_loader
        
        gen_settings = config['generator_settings']
        self.p_clean = gen_settings.get('clean_sample_prob', 0.1)
        self.p_artefact = gen_settings.get('artefact_prob', 0.45)
        self.p_noise = gen_settings.get('ambient_noise_prob', 0.45)
        self.noise_type = noise_type
        self.noise_levels = config.get('noise_level', [-40])

    def __call__(self, clean_audio: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Randomly selects and applies an augmentation to the input audio.
        
        Args:
            clean_audio (np.ndarray): Clean input audio signal.
        
        Returns:
            Tuple[np.ndarray, str]: Augmented audio and a string describing the augmentation type.
        """

        augmentation_options = ['clean', 'artefact', 'noise']
        augmentation_probs = [self.p_clean, self.p_artefact, self.p_noise]
        chosen_augment = random.choices(augmentation_options, weights=augmentation_probs, k=1)[0]
        
        if chosen_augment == 'clean':
            return clean_audio, 'clean'
        
        elif chosen_augment == 'artefact':
            spec = random.choice(self.artefact_specs)
            artefact_aug = ArtefactAugmentation(spec, self.sr, self.duration)
            return artefact_aug(clean_audio), f"artefact_{spec['type']}"
        
        elif chosen_augment == 'noise':
            noise_file = random.choice(self.noise_files)
            noise_audio = self.noise_loader(noise_file, self.sr)
            noise_level = random.choice(self.noise_levels)
            noise_aug = NoiseAugmentation(noise_audio, self.sr, self.duration, noise_level, self.noise_type)
            return noise_aug(clean_audio), f"noise_{noise_file}_{self.noise_type}_{noise_level}dB"
        
        else:
            raise ValueError(f"Unknown augmentation type: {chosen_augment}")