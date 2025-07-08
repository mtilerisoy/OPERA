import numpy as np
from aug.augment.base import BaseAugmentation
from utils import process_audio_length

class NoiseAugmentation(BaseAugmentation):
    """
    Augmentation that adds ambient noise to a clean audio signal at a specified dBFS level.
    """
    def __init__(self, noise_audio: np.ndarray, sr: int, duration: float, background_noise_db_level: float):
        """
        Args:
            noise_audio (np.ndarray): The noise audio signal.
            sr (int): Sample rate.
            duration (float): Target duration in seconds.
            background_noise_db_level (float): Target dBFS for noise.
        """
        self.noise_audio = noise_audio
        self.sr = sr
        self.duration = duration
        self.background_noise_db_level = background_noise_db_level

    def _calculate_rms(self, audio: np.ndarray) -> float:
        """
        Calculates the root mean square (RMS) of an audio signal.
        
        Args:
            audio (np.ndarray): Input audio signal.
        
        Returns:
            float: RMS value.
        """
        return np.sqrt(np.mean(audio**2))

    def _normalize_to_dbfs(self, audio: np.ndarray, target_dbfs: float) -> np.ndarray:
        rms = self._calculate_rms(audio)
        if rms > 0:
            target_rms = 10 ** (target_dbfs / 20.0)
            return audio * (target_rms / rms)
        else:
            return audio

    def __call__(self, clean_audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Adds noise to the clean audio at the specified SNR.
        
        Args:
            clean_audio (np.ndarray): Clean input audio signal.
        
        Returns:
            np.ndarray: Noisy audio signal.
        """
        clean_proc = process_audio_length(clean_audio, self.duration, self.sr)
        noise_proc = process_audio_length(self.noise_audio, self.duration, self.sr)

        # Scale noise to fixed dBFS before mixing
        noise_scaled = self._normalize_to_dbfs(noise_proc, self.background_noise_db_level)
        noisy_audio = clean_proc + noise_scaled
        return noisy_audio 