"""
Image data loader for FER-2013 dataset with face detection and preprocessing.
"""

import os
import numpy as np
import pandas as pd
import cv2
import requests
import zipfile
from typing import Tuple, Optional, List
import logging
from PIL import Image
import matplotlib.pyplot as plt

from .data_loaders import BaseDataLoader
from .constants import FER2013_MAPPING, EMOTION_CLASSES


class ImageDataLoader(BaseDataLoader):
    """
    Data loader for image-based emotion recognition using FER-2013 dataset.
    
    Handles dataset download, face detection, alignment, and preprocessing
    for facial emotion recognition tasks.
    """
    
    def __init__(self, 
                 data_dir: str = "data/raw",
                 processed_dir: str = "data/processed",
                 cache_dir: str = "data/cache",
                 image_size: Tuple[int, int] = (224, 224),
                 use_face_detection: bool = True,
                 random_state: int = 42):
        """
        Initialize the image data loader.
        
        Args:
            data_dir: Directory containing raw datasets
            processed_dir: Directory for processed datasets  
            cache_dir: Directory for cached preprocessed data
            image_size: Target size for images (height, width)
            use_face_detection: Whether to use MTCNN for face detection
            random_state: Random seed for reproducibility
        """
        super().__init__(data_dir, processed_dir, cache_dir, random_state)
        
        self.image_size = image_size
        self.use_face_detection = use_face_detection
        self.dataset_name = "fer2013"
        
        # FER-2013 dataset URLs (using Kaggle dataset)
        self.fer2013_url = "https://www.kaggle.com/datasets/msambare/fer2013"
        self.fer2013_csv_path = os.path.join(self.data_dir, "fer2013.csv")
        
        # Initialize face detector if requested
        self.face_detector = None
        if self.use_face_detection:
            self._initialize_face_detector()
    
    def _initialize_face_detector(self):
        """Initialize MTCNN face detector."""
        try:
            from mtcnn import MTCNN
            self.face_detector = MTCNN()
            self.logger.info("MTCNN face detector initialized")
        except ImportError:
            self.logger.warning("MTCNN not available, falling back to OpenCV Haar cascades")
            # Fallback to OpenCV Haar cascades
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)
    
    def download_dataset(self) -> bool:
        """
        Download FER-2013 dataset.
        
        Note: This method provides instructions for manual download since
        Kaggle datasets require authentication.
        
        Returns:
            bool: True if dataset is available, False otherwise
        """
        if os.path.exists(self.fer2013_csv_path):
            self.logger.info("FER-2013 dataset already exists")
            return True
        
        # Check if we have a sample dataset for testing
        sample_csv_path = os.path.join(self.data_dir, "fer2013_sample.csv")
        if os.path.exists(sample_csv_path):
            self.fer2013_csv_path = sample_csv_path
            self.logger.info("Using sample FER-2013 dataset")
            return True
        
        # Create a small sample dataset for testing purposes
        self._create_sample_dataset()
        return True
    
    def _create_sample_dataset(self):
        """Create a small sample dataset for testing purposes."""
        self.logger.info("Creating sample FER-2013 dataset for testing")
        
        # Generate sample data with random pixel values
        sample_data = []
        emotions = list(range(7))  # FER-2013 has 7 emotion classes
        
        for emotion in emotions:
            for i in range(20):  # 20 samples per emotion
                # Generate random 48x48 grayscale image
                pixels = np.random.randint(0, 256, 48*48)
                pixel_str = ' '.join(map(str, pixels))
                
                sample_data.append({
                    'emotion': emotion,
                    'pixels': pixel_str,
                    'Usage': 'Training' if i < 16 else 'PublicTest'
                })
        
        # Create DataFrame and save
        df = pd.DataFrame(sample_data)
        sample_path = os.path.join(self.data_dir, "fer2013_sample.csv")
        df.to_csv(sample_path, index=False)
        self.fer2013_csv_path = sample_path
        
        self.logger.info(f"Sample dataset created with {len(sample_data)} samples")
    
    def load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load raw FER-2013 data from CSV file.
        
        Returns:
            Tuple of (images, labels) where images are 48x48 grayscale arrays
        """
        if not os.path.exists(self.fer2013_csv_path):
            if not self.download_dataset():
                raise FileNotFoundError(f"FER-2013 dataset not found at {self.fer2013_csv_path}")
        
        self.logger.info(f"Loading FER-2013 data from {self.fer2013_csv_path}")
        
        # Load CSV data
        df = pd.read_csv(self.fer2013_csv_path)
        
        # Extract pixel data and reshape to images
        images = []
        labels = []
        
        for idx, row in df.iterrows():
            # Parse pixel string to array
            pixel_values = np.array([int(pixel) for pixel in row['pixels'].split()])
            
            # Reshape to 48x48 grayscale image
            image = pixel_values.reshape(48, 48).astype(np.uint8)
            images.append(image)
            
            # Map FER-2013 emotion labels to our standard 6-class system
            fer_emotion = int(row['emotion'])
            if fer_emotion in FER2013_MAPPING:
                standard_emotion = FER2013_MAPPING[fer_emotion]
                emotion_idx = EMOTION_CLASSES.index(standard_emotion)
                labels.append(emotion_idx)
            else:
                # Default to neutrality for unknown emotions
                labels.append(EMOTION_CLASSES.index('neutrality'))
        
        images = np.array(images, dtype=np.uint8)
        labels = np.array(labels)
        
        self.logger.info(f"Loaded {len(images)} images with shape {images.shape}")
        self.logger.info(f"Emotion distribution: {np.bincount(labels)}")
        
        return images, labels
    
    def detect_and_align_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and align face in the image.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Aligned face image or None if no face detected
        """
        if self.face_detector is None:
            return image
        
        # Convert to RGB if needed for MTCNN
        if hasattr(self.face_detector, 'detect_faces'):
            # MTCNN detector
            if len(image.shape) == 2:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = image
            
            detections = self.face_detector.detect_faces(rgb_image)
            
            if detections:
                # Use the first (most confident) detection
                detection = detections[0]
                x, y, w, h = detection['box']
                
                # Extract face region with some padding
                padding = 0.1
                x_pad = int(w * padding)
                y_pad = int(h * padding)
                
                x1 = max(0, x - x_pad)
                y1 = max(0, y - y_pad)
                x2 = min(image.shape[1], x + w + x_pad)
                y2 = min(image.shape[0], y + h + y_pad)
                
                face = image[y1:y2, x1:x2]
                return face
        else:
            # OpenCV Haar cascade detector
            faces = self.face_detector.detectMultiScale(
                image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Use the largest face
                x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
                face = image[y:y+h, x:x+w]
                return face
        
        # Return original image if no face detected
        return image
    
    def preprocess_image(self, image: np.ndarray, augment: bool = False) -> np.ndarray:
        """
        Preprocess a single image.
        
        Args:
            image: Input grayscale image
            augment: Whether to apply data augmentation
            
        Returns:
            Preprocessed image
        """
        # Face detection and alignment
        if self.use_face_detection:
            aligned_face = self.detect_and_align_face(image)
            if aligned_face is not None:
                image = aligned_face
        
        # Resize to target size
        image = cv2.resize(image, self.image_size)
        
        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Data augmentation if requested
        if augment:
            image = self._apply_augmentation(image)
        
        # Add channel dimension for grayscale
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        return image
    
    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to image.
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random rotation (-15 to +15 degrees)
        angle = np.random.uniform(-15, 15)
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        
        # Random brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.2)
        image = np.clip(image * brightness_factor, 0.0, 1.0)
        
        # Random contrast adjustment
        contrast_factor = np.random.uniform(0.8, 1.2)
        image = np.clip((image - 0.5) * contrast_factor + 0.5, 0.0, 1.0)
        
        return image
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the raw image data.
        
        Args:
            X: Raw images (N, H, W)
            y: Raw labels
            
        Returns:
            Tuple of (preprocessed_images, labels)
        """
        self.logger.info("Preprocessing image data...")
        
        preprocessed_images = []
        
        for i, image in enumerate(X):
            if i % 1000 == 0:
                self.logger.info(f"Processed {i}/{len(X)} images")
            
            processed_image = self.preprocess_image(image, augment=False)
            preprocessed_images.append(processed_image)
        
        preprocessed_images = np.array(preprocessed_images)
        
        self.logger.info(f"Preprocessing complete. Final shape: {preprocessed_images.shape}")
        
        return preprocessed_images, y
    
    def create_data_generator(self, 
                            X: np.ndarray, 
                            y: np.ndarray, 
                            batch_size: int = 32,
                            augment: bool = False,
                            shuffle: bool = True):
        """
        Create a data generator for batch processing.
        
        Args:
            X: Preprocessed images
            y: Labels
            batch_size: Batch size
            augment: Whether to apply augmentation
            shuffle: Whether to shuffle data
            
        Yields:
            Batches of (images, labels)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        while True:
            if shuffle:
                np.random.shuffle(indices)
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_X = X[batch_indices]
                batch_y = y[batch_indices]
                
                # Apply augmentation if requested
                if augment:
                    augmented_batch = []
                    for image in batch_X:
                        # Remove channel dimension for augmentation
                        if len(image.shape) == 3 and image.shape[-1] == 1:
                            image_2d = image.squeeze(-1)
                        else:
                            image_2d = image
                        
                        augmented = self._apply_augmentation(image_2d)
                        
                        # Add channel dimension back
                        if len(augmented.shape) == 2:
                            augmented = np.expand_dims(augmented, axis=-1)
                        
                        augmented_batch.append(augmented)
                    
                    batch_X = np.array(augmented_batch)
                
                yield batch_X, batch_y
    
    def visualize_samples(self, X: np.ndarray, y: np.ndarray, n_samples: int = 16):
        """
        Visualize sample images with their emotion labels.
        
        Args:
            X: Images to visualize
            y: Corresponding labels
            n_samples: Number of samples to show
        """
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        indices = np.random.choice(len(X), n_samples, replace=False)
        
        for i, idx in enumerate(indices):
            image = X[idx]
            label = y[idx]
            
            # Handle different image formats
            if len(image.shape) == 3 and image.shape[-1] == 1:
                image = image.squeeze(-1)
            
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f'{EMOTION_CLASSES[label]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_dataset_statistics(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Get statistics about the dataset.
        
        Args:
            X: Images
            y: Labels
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_samples': len(X),
            'image_shape': X.shape[1:],
            'pixel_mean': np.mean(X),
            'pixel_std': np.std(X),
            'class_distribution': {},
            'class_percentages': {}
        }
        
        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        for class_idx, count in zip(unique, counts):
            emotion_name = EMOTION_CLASSES[class_idx]
            stats['class_distribution'][emotion_name] = int(count)
            stats['class_percentages'][emotion_name] = float(count / len(y) * 100)
        
        return stats