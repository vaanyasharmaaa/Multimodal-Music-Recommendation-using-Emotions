"""
Base data loader class and common functionality for multi-modal emotion detection.
"""

import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict, Any
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import logging

from .constants import EMOTION_MAPPING, INDEX_TO_EMOTION, EMOTION_CLASSES, NUM_CLASSES


class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders with common functionality.
    
    This class provides common methods for data loading, preprocessing,
    and train/validation/test splitting with stratification.
    """
    
    def __init__(self, 
                 data_dir: str = "data/raw",
                 processed_dir: str = "data/processed", 
                 cache_dir: str = "data/cache",
                 random_state: int = 42):
        """
        Initialize the base data loader.
        
        Args:
            data_dir: Directory containing raw datasets
            processed_dir: Directory for processed datasets
            cache_dir: Directory for cached preprocessed data
            random_state: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.cache_dir = cache_dir
        self.random_state = random_state
        
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Data storage
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
    @abstractmethod
    def download_dataset(self) -> bool:
        """
        Download the dataset if not already present.
        
        Returns:
            bool: True if download successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load_raw_data(self) -> Tuple[Any, Any]:
        """
        Load raw data from the dataset.
        
        Returns:
            Tuple containing features and labels
        """
        pass
    
    @abstractmethod
    def preprocess_data(self, X: Any, y: Any) -> Tuple[Any, Any]:
        """
        Preprocess the raw data.
        
        Args:
            X: Raw features
            y: Raw labels
            
        Returns:
            Tuple containing preprocessed features and labels
        """
        pass
    
    def normalize_emotion_labels(self, labels: List[str]) -> np.ndarray:
        """
        Normalize emotion labels to the standard 6-class system.
        
        Args:
            labels: List of emotion labels (strings or integers)
            
        Returns:
            numpy array of normalized emotion indices
        """
        normalized_labels = []
        
        for label in labels:
            # Convert to string and lowercase for consistent mapping
            if isinstance(label, (int, np.integer)):
                # Handle numeric labels (dataset-specific mapping needed)
                label_str = str(label)
            else:
                label_str = str(label).lower().strip()
            
            # Map to standard emotion
            if label_str in EMOTION_MAPPING:
                normalized_labels.append(EMOTION_MAPPING[label_str])
            else:
                # Default to neutrality for unknown emotions
                self.logger.warning(f"Unknown emotion label: {label_str}, mapping to neutrality")
                normalized_labels.append(EMOTION_MAPPING['neutrality'])
        
        return np.array(normalized_labels)
    
    def convert_labels_to_categorical(self, labels: np.ndarray) -> np.ndarray:
        """
        Convert emotion indices to one-hot encoded categorical format.
        
        Args:
            labels: Array of emotion indices
            
        Returns:
            One-hot encoded labels
        """
        try:
            from tensorflow.keras.utils import to_categorical
            return to_categorical(labels, num_classes=NUM_CLASSES)
        except ImportError:
            # Fallback implementation if TensorFlow is not available
            categorical = np.zeros((len(labels), NUM_CLASSES))
            categorical[np.arange(len(labels)), labels] = 1
            return categorical
    
    def create_stratified_splits(self, 
                               X: Any, 
                               y: np.ndarray,
                               test_size: float = 0.2,
                               val_size: float = 0.2) -> Tuple[Any, Any, Any, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create stratified train/validation/test splits.
        
        Args:
            X: Features
            y: Labels (emotion indices)
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation set
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            stratify=y, 
            random_state=self.random_state
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=self.random_state
        )
        
        # Store splits
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        
        # Log split information
        self.logger.info(f"Data splits created:")
        self.logger.info(f"  Train: {len(X_train)} samples")
        self.logger.info(f"  Validation: {len(X_val)} samples") 
        self.logger.info(f"  Test: {len(X_test)} samples")
        
        # Log class distribution
        self._log_class_distribution(y_train, "Train")
        self._log_class_distribution(y_val, "Validation")
        self._log_class_distribution(y_test, "Test")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _log_class_distribution(self, labels: np.ndarray, split_name: str):
        """Log the class distribution for a data split."""
        unique, counts = np.unique(labels, return_counts=True)
        self.logger.info(f"  {split_name} class distribution:")
        for emotion_idx, count in zip(unique, counts):
            emotion_name = INDEX_TO_EMOTION.get(emotion_idx, f"Unknown({emotion_idx})")
            percentage = (count / len(labels)) * 100
            self.logger.info(f"    {emotion_name}: {count} ({percentage:.1f}%)")
    
    def get_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for handling imbalanced datasets.
        
        Args:
            labels: Array of emotion indices
            
        Returns:
            Dictionary mapping class indices to weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(labels)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=labels
        )
        
        class_weights = dict(zip(classes, weights))
        
        self.logger.info("Class weights calculated:")
        for class_idx, weight in class_weights.items():
            emotion_name = INDEX_TO_EMOTION.get(class_idx, f"Unknown({class_idx})")
            self.logger.info(f"  {emotion_name}: {weight:.3f}")
            
        return class_weights
    
    def save_processed_data(self, filename: str):
        """
        Save processed data splits to disk.
        
        Args:
            filename: Base filename for saving (without extension)
        """
        if self.X_train is None:
            raise ValueError("No processed data to save. Run create_stratified_splits first.")
        
        save_path = os.path.join(self.processed_dir, filename)
        
        np.savez_compressed(
            f"{save_path}.npz",
            X_train=self.X_train,
            X_val=self.X_val, 
            X_test=self.X_test,
            y_train=self.y_train,
            y_val=self.y_val,
            y_test=self.y_test
        )
        
        self.logger.info(f"Processed data saved to {save_path}.npz")
    
    def load_processed_data(self, filename: str) -> bool:
        """
        Load processed data splits from disk.
        
        Args:
            filename: Base filename to load (without extension)
            
        Returns:
            bool: True if loading successful, False otherwise
        """
        load_path = os.path.join(self.processed_dir, f"{filename}.npz")
        
        if not os.path.exists(load_path):
            self.logger.warning(f"Processed data file not found: {load_path}")
            return False
        
        try:
            data = np.load(load_path, allow_pickle=True)
            self.X_train = data['X_train']
            self.X_val = data['X_val']
            self.X_test = data['X_test']
            self.y_train = data['y_train']
            self.y_val = data['y_val']
            self.y_test = data['y_test']
            
            self.logger.info(f"Processed data loaded from {load_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading processed data: {e}")
            return False
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded data.
        
        Returns:
            Dictionary containing data information
        """
        if self.X_train is None:
            return {"status": "No data loaded"}
        
        info = {
            "train_samples": len(self.X_train),
            "val_samples": len(self.X_val),
            "test_samples": len(self.X_test),
            "num_classes": NUM_CLASSES,
            "emotion_classes": EMOTION_CLASSES,
            "train_shape": getattr(self.X_train, 'shape', 'N/A'),
            "class_distribution": {}
        }
        
        # Add class distribution for each split
        for split_name, labels in [("train", self.y_train), ("val", self.y_val), ("test", self.y_test)]:
            unique, counts = np.unique(labels, return_counts=True)
            info["class_distribution"][split_name] = {
                INDEX_TO_EMOTION.get(idx, f"Unknown({idx})"): int(count) 
                for idx, count in zip(unique, counts)
            }
        
        return info


def emotion_label_converter(labels: List[Any], dataset_mapping: Optional[Dict] = None) -> np.ndarray:
    """
    Utility function to convert various emotion label formats to standardized indices.
    
    Args:
        labels: List of emotion labels (strings, integers, etc.)
        dataset_mapping: Optional dataset-specific mapping dictionary
        
    Returns:
        numpy array of standardized emotion indices
    """
    if dataset_mapping:
        # Apply dataset-specific mapping first
        mapped_labels = []
        for label in labels:
            if label in dataset_mapping:
                mapped_labels.append(dataset_mapping[label])
            else:
                mapped_labels.append(label)
        labels = mapped_labels
    
    # Direct implementation without instantiating abstract class
    normalized_labels = []
    
    for label in labels:
        # Convert to string and lowercase for consistent mapping
        if isinstance(label, (int, np.integer)):
            # Handle numeric labels (dataset-specific mapping needed)
            label_str = str(label)
        else:
            label_str = str(label).lower().strip()
        
        # Map to standard emotion
        if label_str in EMOTION_MAPPING:
            normalized_labels.append(EMOTION_MAPPING[label_str])
        else:
            # Default to neutrality for unknown emotions
            normalized_labels.append(EMOTION_MAPPING['neutrality'])
    
    return np.array(normalized_labels)


def validate_emotion_distribution(labels: np.ndarray, min_samples_per_class: int = 10) -> bool:
    """
    Validate that the emotion distribution is suitable for training.
    
    Args:
        labels: Array of emotion indices
        min_samples_per_class: Minimum number of samples required per class
        
    Returns:
        bool: True if distribution is valid, False otherwise
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    # Check if all emotion classes are present
    missing_classes = set(range(NUM_CLASSES)) - set(unique)
    if missing_classes:
        logging.warning(f"Missing emotion classes: {[INDEX_TO_EMOTION[idx] for idx in missing_classes]}")
        return False
    
    # Check minimum samples per class
    insufficient_classes = []
    for class_idx, count in zip(unique, counts):
        if count < min_samples_per_class:
            insufficient_classes.append((INDEX_TO_EMOTION[class_idx], count))
    
    if insufficient_classes:
        logging.warning(f"Classes with insufficient samples: {insufficient_classes}")
        return False
    
    return True