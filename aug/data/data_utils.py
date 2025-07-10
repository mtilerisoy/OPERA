import os
import pandas as pd
import numpy as np
from typing import List, Tuple

def list_wav_files(directory: str) -> List[str]:
    """
    Lists all .wav files in a directory.
    
    Args:
        directory (str): Directory to search for .wav files.
    
    Returns:
        List[str]: List of .wav filenames in the directory.
    """

    return [f for f in os.listdir(directory) if f.endswith('.wav')]

def modify_sound_dir_loc(sound_dir_loc: np.ndarray, clean_data_path: str) -> np.ndarray:
    """
    Modifies the sound_dir_loc to point to the clean_data_path.
    """
    modified_paths = []
    for path in sound_dir_loc:
        filename = os.path.basename(path)
        modified_path = os.path.join(clean_data_path, filename)
        modified_paths.append(modified_path)
    return np.array(modified_paths)

def load_dataset_from_npy(config: dict, split: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Loads dataset information from preprocessed .npy files using explicit file names from config.
    
    Args:
        feature_dir (str): Directory containing the .npy files (e.g., 'feature/coughvid_eval/')
        config (dict): Configuration dictionary containing file names
        split (str): Data split to filter by ('train', 'val', 'test', or 'all' for all splits)
    
    Returns:
        Tuple[List[str], List[int], List[str]]: 
            - List of audio file paths (modified to point to clean_data_path)
            - List of labels
            - List of split assignments
    """
    
    # Load the .npy files using explicit file names from config
    feature_dir = config['feature_dir']
    sound_dir_loc = np.load(os.path.join(feature_dir, config['sound_dir_loc_file']), allow_pickle=True)
    labels = np.load(os.path.join(feature_dir, config['labels_file']), allow_pickle=True)
    splits = np.load(os.path.join(feature_dir, config['split_file']), allow_pickle=True)
    
    # Extract base filenames and prepend clean_data_path
    clean_data_path = config['clean_data_path']
    if clean_data_path:
        sound_dir_loc = modify_sound_dir_loc(sound_dir_loc, clean_data_path)
    
    # Filter by split (or return all if split is 'all')
    if split == 'all':
        return sound_dir_loc.tolist(), labels.tolist(), splits.tolist()
    else:
        split_mask = np.array(splits) == split
        filtered_paths = sound_dir_loc[split_mask]
        filtered_labels = labels[split_mask]
        filtered_splits = np.array(splits)[split_mask]
        
        return filtered_paths.tolist(), filtered_labels.tolist(), filtered_splits.tolist()

def validate_dataset_files(feature_dir: str, config: dict) -> bool:
    """
    Validates that all required .npy files exist for a dataset using explicit file names from config.
    
    Args:
        feature_dir (str): Directory containing the .npy files
        config (dict): Configuration dictionary containing file names
    
    Returns:
        bool: True if all required files exist, False otherwise
    """
    required_files = [
        config['sound_dir_loc_file'],
        config['labels_file'],
        config['split_file']
    ]
    
    for file in required_files:
        file_path = os.path.join(feature_dir, file)
        if not os.path.exists(file_path):
            print(f"Missing required file: {file_path}")
            return False
    
    return True 