import os
import torch
from aug.config import Config
from aug.data.dataset import NoisyICBHIDataGenerator
from utils import set_deterministic, save_audio


def generate_entire_augmented_dataset(config_path: str, output_dir: str, split: str = 'train'):
    """
    Generates the entire augmented audio dataset for a given split and saves to disk.
    
    Args:
        config_path (str): Path to the YAML config file.
        output_dir (str): Directory to save the generated audio files.
        split (str): Data split to generate ('train' or 'test').
    """
    print(f"--- Generating Entire Audio Dataset for '{split}' split ---")
    
    config = Config(config_path)
    dataset = NoisyICBHIDataGenerator(config, split, debug=True)
    
    if len(dataset) == 0:
        print(f"[WARNING] The generator for the '{split}' split contains 0 files. Cannot proceed with generation.")
        return
    print(f"\nGenerator instantiated successfully. Found {len(dataset)} files.")
    
    # output_identifier = config['background_noise_level_db'][0]
    # output_dir = output_dir + "__" + str(output_identifier) + "dB"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Audio files will be saved in: '{output_dir}/'")
    
    for i in range(len(dataset)):
        data_tensor, _, aug_type = dataset[i]
        file_name = dataset.file_list[i]
        data_tensor = data_tensor.numpy()
        sr = config['sample_rate']
        save_audio(os.path.join(output_dir, f"{file_name}.wav"), data_tensor, sr)
        if i % 100 == 0:
            print(f"Saved {i} samples so far... (last aug: {aug_type})")
    
    print("\n--- Full Audio Dataset Generation Complete ---")

if __name__ == '__main__':
    """
    Main entry point for generating the full augmented dataset for both train and test splits.
    """
    config_path = 'config.yaml'
    config = Config(config_path)
    output_dir = config['output_path']

    # set_deterministic(config['random_seed'])
    
    generate_entire_augmented_dataset(config_path, output_dir, split='train')
    generate_entire_augmented_dataset(config_path, output_dir, split='test') 