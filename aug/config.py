import yaml
from typing import Any, Dict

class Config:
    """
    Loads and validates configuration from a YAML file for the augmentation pipeline.
    Provides dictionary-like access to config values.
    """
    def __init__(self, config_path: str):
        """
        Initialize Config by loading and validating the YAML config file.
        
        Args:
            config_path (str): Path to the YAML configuration file.
        """

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        self._validate()

    def _validate(self):
        """
        Validates that all required configuration keys are present.
        
        Raises:
            ValueError: If any required key is missing.
        """

        # Check if using generic dataset approach
        if self._config.get('use_generic_dataset', False):
            # Generic dataset mode - requires different parameters
            required_keys = [
                'dataset_name', 'sound_dir_loc_file', 'labels_file', 'split_file', 'clean_data_path', 'noise_data_path',
                'output_path', 'sample_rate', 'target_duration_sec',
                'noise_level', 'generator_settings', 'random_seed', 'noise_type'
            ]
        else:
            # Legacy ICBHI-specific mode
            required_keys = [
                'clean_data_path', 'clean_data_split_path', 'noise_data_path',
                'output_path', 'sample_rate', 'target_duration_sec',
                'noise_level', 'generator_settings', 'random_seed', 'noise_type'
            ]

        for key in required_keys:
            if key not in self._config:
                raise ValueError(f"Missing required config key: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with an optional default.
        
        Args:
            key (str): The configuration key.
            default (Any): Default value if key is not found.
        
        Returns:
            Any: The configuration value or default.
        """

        return self._config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """
        Dictionary-style access to configuration values.
        
        Args:
            key (str): The configuration key.
        
        Returns:
            Any: The configuration value.
        """

        return self._config[key]

    @property
    def dict(self) -> Dict[str, Any]:
        """
        Returns the entire configuration as a dictionary.
        
        Returns:
            Dict[str, Any]: The configuration dictionary.
        """
        
        return self._config 