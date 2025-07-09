import os
import numpy as np
import torch
from torch.utils.data import Dataset
from aug.utils import load_audio, process_audio_length, pre_process_audio_mel, peak_normalize
from aug.data.data_utils import load_split_file_list, list_wav_files
from aug.augment.pipeline import AugmentationPipeline

class NoisyICBHIDataGenerator(Dataset):
    """
    PyTorch Dataset for generating noisy/augmented ICBHI audio samples on-the-fly.

    Each sample is loaded from the ICBHI dataset, optionally augmented with either
    synthetic artefacts or noise from the ESC-50 dataset, or left clean, according to
    probabilities specified in the configuration. The dataset can be used for both
    training and evaluation, and supports debug mode for easier inspection.
    """
    def __init__(self, config, split: str, debug: bool = False):
        """
        Initialize the dataset.

        Args:
            config: Configuration object or dictionary with all pipeline settings.
            split (str): Data split to use ('train' or 'test').
            debug (bool): If True, disables mel conversion and uses peak normalization for easier inspection.
        """
        super().__init__()
        
        self.config = config
        self.debug = debug

        self.sr = config['sample_rate']
        self.duration = config['target_duration_sec']
        self.clean_data_path = config['clean_data_path']
        self.noise_data_path = config['noise_data_path']
        
        self.file_list = load_split_file_list(config['clean_data_split_path'], split)
        self.file_list.sort()
        
        self.noise_files = list_wav_files(self.noise_data_path)
        self.noise_files.sort()

        self.noise_type = config['noise_type']
        
        self.artefact_specs = [
            {'type': 'low_pass', 'cutoff_hz': 2000, 'order': 5},
            {'type': 'low_pass', 'cutoff_hz': 4000, 'order': 5},
            {'type': 'clipping', 'gain_db': 6.0},
            {'type': 'clipping', 'gain_db': 12.0},
        ]
        
        self.pipeline = AugmentationPipeline(
            config,
            self.noise_files,
            self.artefact_specs,
            self.sr,
            self.duration,
            self._load_noise_audio,
            self.noise_type
        )

    def _load_noise_audio(self, filename, sr):
        """
        Loads a noise audio file by filename.

        Args:
            filename (str): Name of the noise file.
            sr (int): Sample rate.
        Returns:
            np.ndarray: Loaded noise audio signal, or None if loading fails.
        """
        path = os.path.join(self.noise_data_path, filename)
        return load_audio(path, sr)

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Loads and augments a sample by index.

        Args:
            idx (int): Index of the sample to load.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, str]:
                - Augmented audio (torch.Tensor)
                - Clean audio (torch.Tensor)
                - Augmentation type (str)
        Notes:
            - If debug is True, returns peak-normalized waveforms.
            - If debug is False, returns mel-spectrograms.
            - If the audio file is missing, returns silent tensors.
        """
        csv_filename = self.file_list[idx]
        base_name = os.path.splitext(csv_filename)[0]
        load_filename = f"{base_name}.wav"
        clean_path = os.path.join(self.clean_data_path, load_filename)
        clean_audio_raw = load_audio(clean_path, sr=self.sr)
        
        if clean_audio_raw is None:
            silent_len = int(self.duration * self.sr)
            silent_audio = np.zeros(silent_len, dtype=np.float32)
            return torch.from_numpy(silent_audio), torch.from_numpy(silent_audio), "silent"
            
        clean_audio_processed = process_audio_length(clean_audio_raw, self.duration, self.sr)
        aug_audio, aug_type = self.pipeline(clean_audio_processed)
        
        if self.debug:
            aug_audio = peak_normalize(aug_audio)
            clean_audio_processed = peak_normalize(clean_audio_processed)
            return torch.from_numpy(aug_audio), torch.from_numpy(clean_audio_processed), aug_type
        
        aug_audio = pre_process_audio_mel(aug_audio, f_max=8000, sr=self.sr)
        clean_audio_processed = pre_process_audio_mel(clean_audio_processed, f_max=8000, sr=self.sr)
        return torch.from_numpy(aug_audio), torch.from_numpy(clean_audio_processed), aug_type 