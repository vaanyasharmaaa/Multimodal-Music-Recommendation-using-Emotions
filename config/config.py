"""
Configuration management for multi-modal emotion detection system.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration manager for the emotion detection system."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.model_config = self._load_yaml("model_config.yaml")
        self.dataset_config = self._load_yaml("dataset_config.yaml")
        
        # Set up paths
        self.setup_paths()
        
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        config_path = self.config_dir / filename
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {config_path}: {e}")
    
    def setup_paths(self):
        """Create necessary directories if they don't exist."""
        directories = [
            "data/raw",
            "data/processed", 
            "data/splits",
            "data/cache",
            "models/image",
            "models/text",
            "models/video", 
            "models/multimodal",
            "models/checkpoints",
            "notebooks",
            "utils"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for specific model type."""
        if model_type not in ['image_model', 'text_model', 'video_model', 'multimodal_model']:
            raise ValueError(f"Unknown model type: {model_type}")
        return self.model_config.get(model_type, {})
    
    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """Get configuration for specific dataset."""
        datasets = self.dataset_config.get('datasets', {})
        if dataset_name not in datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return datasets[dataset_name]
    
    def get_emotion_mapping(self) -> Dict[str, int]:
        """Get the standard emotion label mapping."""
        labels = self.model_config.get('emotion_labels', [])
        return {label: idx for idx, label in enumerate(labels)}
    
    def get_paths(self) -> Dict[str, str]:
        """Get all important paths."""
        return {
            'data_raw': 'data/raw',
            'data_processed': 'data/processed',
            'data_splits': 'data/splits', 
            'data_cache': 'data/cache',
            'models': 'models',
            'notebooks': 'notebooks',
            'config': str(self.config_dir)
        }

# Global configuration instance
config = Config()