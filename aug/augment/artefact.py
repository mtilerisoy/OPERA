import numpy as np
import augly.audio as A
from aug.augment.base import BaseAugmentation
from aug.utils import process_audio_length

class ArtefactAugmentation(BaseAugmentation):
    """
    Augmentation that applies a composed pipeline of AugLy audio augmentations with explicit probabilities.
    Returns both the augmented audio and a dict of applied transforms and their intensities.
    """
    def __init__(self, artefact_spec: dict, sr: int, duration: float):
        """
        Args:
            artefact_spec (dict): Not used, for compatibility.
            sr (int): Sample rate.
            duration (float): Target duration in seconds.
        """
        self.sr = sr
        self.duration = duration
        # Define the pipeline with explicit probabilities
        # Probabilities are commented next to each augmentation

        self.effects = A.Compose([
            A.ChangeVolume(volume_db=5,                               p=1.0),
            # A.Clicks(                                                 p=1.0),
            # A.Harmonic(                                                 p=1.0),
            # A.Speed(factor=50,                                       p=1.0),
            # A.PitchShift(n_steps=5,                                 p=1.0),
            # A.Reverb(reverberance=100, hf_damping=1, pre_delay=1.0, wet_gain=3.0,                                  p=1.0),
            # A.LowPassFilter(cutoff_hz=20,                                 p=1.0),
        ])
    
    def estimate_snr(self, original_audio: np.ndarray, distorted_audio: np.ndarray) -> float:
        """
        Estimate the Signal-to-Noise Ratio (SNR) between original and distorted audio.
        
        SNR is calculated as: SNR = 10 * log10(signal_power / noise_power)
        where signal_power is the power of the original signal and noise_power is the power of the distortion.
        
        Args:
            original_audio (np.ndarray): Original clean audio signal
            distorted_audio (np.ndarray): Distorted audio signal after augmentation
            
        Returns:
            float: SNR in decibels (dB)
        """
        # Ensure both signals have the same length
        min_length = min(len(original_audio), len(distorted_audio))
        original_audio = original_audio[:min_length]
        distorted_audio = distorted_audio[:min_length]
        
        # Calculate signal power (original audio)
        signal_power = np.mean(original_audio ** 2)
        
        # Calculate the power of the injected noise
        noise = original_audio - distorted_audio
        noise_power = np.mean(noise ** 2)
        
        # perfectly clean signal - no noise
        if noise_power == 0:
            return float('inf')
        
        # Calculate SNR in dB
        snr_db = 10 * np.log10(signal_power / noise_power)
        
        return snr_db

    def __call__(self, clean_audio: np.ndarray, **kwargs):
        # Store original audio for SNR calculation
        original_audio = clean_audio.copy()
        
        # Apply augmentation pipeline
        noisy_audio, _ = self.effects(clean_audio, sample_rate=self.sr)
        
        # Estimate SNR
        snr_db = self.estimate_snr(original_audio, noisy_audio)
        
        return noisy_audio, snr_db