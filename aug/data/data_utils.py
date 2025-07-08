import os
import pandas as pd
from typing import List

def load_split_file_list(split_csv_path: str, split: str) -> List[str]:
    """
    Loads a list of audio filenames for a given split from a CSV file.
    
    Args:
        split_csv_path (str): Path to the CSV file containing split information.
        split (str): The split to filter by (e.g., 'train', 'test').
    
    Returns:
        List[str]: List of audio filenames for the specified split.
    """
    
    df_splits = pd.read_csv(split_csv_path)
    return df_splits[df_splits['split'] == split]['audio_filename'].tolist()

def list_wav_files(directory: str) -> List[str]:
    """
    Lists all .wav files in a directory.
    
    Args:
        directory (str): Directory to search for .wav files.
    
    Returns:
        List[str]: List of .wav filenames in the directory.
    """

    return [f for f in os.listdir(directory) if f.endswith('.wav')] 