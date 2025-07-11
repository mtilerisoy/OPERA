import os
from aug.config import Config
from aug.data.dataset import NoisyICBHIDataGenerator
from aug.utils import set_deterministic, save_audio
import argparse

def generate_entire_augmented_dataset(config: Config, split: str = 'train'):
    """
    Generates the entire augmented audio dataset for a given split and saves to disk.
    
    Args:
        config_path (str): Path to the YAML config file.
        output_dir (str): Directory to save the generated audio files.
        split (str): Data split to generate ('train' or 'test').
    """
    print(f"--- Generating Entire Audio Dataset for '{split}' split ---")
    
    dataset = NoisyICBHIDataGenerator(config, split, debug=True)
    output_dir = config['output_path']
    
    if len(dataset) == 0:
        print(f"[WARNING] The generator for the '{split}' split contains 0 files. Cannot proceed with generation.")
        return
    print(f"\nGenerator instantiated successfully. Found {len(dataset)} files.")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Audio files will be saved in: '{output_dir}/'")
    
    for i in range(len(dataset)):
        data_tensor, _, aug_type = dataset[i]
        file_name = os.path.basename(dataset.file_paths[i])
        data_tensor = data_tensor.numpy()
        sr = config['sample_rate']
        save_audio(os.path.join(output_dir, f"{file_name}"), data_tensor, sr)
        if i % 100 == 0:
            print(f"Saved {i} samples so far... (last aug: {aug_type})")
    
    print("\n--- Full Audio Dataset Generation Complete ---")

if __name__ == '__main__':
    """
    Main entry point for generating the full augmented dataset for both train and test splits.
    """
    parser = argparse.ArgumentParser(description='Generate augmented datasets using preprocessed .npy files')
    parser.add_argument('--config', type=str, default='config_icbhidisease_generic.yaml', required=True, help='Path to YAML config file')
    # parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'], 
    #                    help='Data split to generate (default: train)')
    
    args = parser.parse_args()
    
    
    # Set deterministic behavior
    config = Config(args.config)
    set_deterministic(config['random_seed'])
    print(f"##### [INFO] Generating dataset with noise level: {config['noise_level']} #####")
    
    generate_entire_augmented_dataset(config, split='train')
    generate_entire_augmented_dataset(config, split='test')
    try:
        generate_entire_augmented_dataset(config, split='val')
    except Exception as e:
        print(f"[DEBUG] No val split found with error: {e}")