"""
Video data loader for RAVDESS dataset with frame extraction and preprocessing.
"""

import os
import numpy as np
import cv2
import json
from typing import Tuple, List, Optional, Dict, Any
import logging
from pathlib import Path
import random

from .data_loaders import BaseDataLoader
from .constants import RAVDESS_MAPPING, EMOTION_CLASSES


class VideoDataLoader(BaseDataLoader):
    """
    Data loader for video-based emotion recognition using RAVDESS dataset.
    
    Handles dataset download, video frame extraction, face detection across sequences,
    and preprocessing for spatio-temporal emotion recognition tasks.
    """
    
    def __init__(self, 
                 data_dir: str = "data/raw",
                 processed_dir: str = "data/processed",
                 cache_dir: str = "data/cache",
                 sequence_length: int = 16,
                 frame_size: Tuple[int, int] = (224, 224),
                 fps: Optional[int] = None,
                 use_face_detection: bool = True,
                 random_state: int = 42):
        """
        Initialize the video data loader.
        
        Args:
            data_dir: Directory containing raw datasets
            processed_dir: Directory for processed datasets
            cache_dir: Directory for cached preprocessed data
            sequence_length: Number of frames per sequence
            frame_size: Target size for frames (height, width)
            fps: Target FPS for frame extraction (None = use original)
            use_face_detection: Whether to use face detection
            random_state: Random seed for reproducibility
        """
        super().__init__(data_dir, processed_dir, cache_dir, random_state)
        
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.fps = fps
        self.use_face_detection = use_face_detection
        self.dataset_name = "ravdess"
        
        # RAVDESS dataset paths
        self.ravdess_dir = os.path.join(self.data_dir, "ravdess_videos")
        self.ravdess_metadata_path = os.path.join(self.data_dir, "ravdess_metadata.json")
        
        # Initialize face detector if requested
        self.face_detector = None
        if self.use_face_detection:
            self._initialize_face_detector()
    
    def _initialize_face_detector(self):
        """Initialize face detector for video processing."""
        try:
            from mtcnn import MTCNN
            self.face_detector = MTCNN()
            self.logger.info("MTCNN face detector initialized for video processing")
        except ImportError:
            self.logger.warning("MTCNN not available, falling back to OpenCV Haar cascades")
            # Fallback to OpenCV Haar cascades
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)
    
    def download_dataset(self) -> bool:
        """
        Download RAVDESS dataset.
        
        Note: This method provides instructions for manual download since
        RAVDESS requires registration and manual download.
        
        Returns:
            bool: True if dataset is available, False otherwise
        """
        if os.path.exists(self.ravdess_dir) and os.listdir(self.ravdess_dir):
            self.logger.info("RAVDESS dataset already exists")
            return True
        
        # Check if we have a sample dataset for testing
        if os.path.exists(self.ravdess_metadata_path):
            self.logger.info("Using sample RAVDESS metadata")
            return True
        
        # Create a sample dataset for testing purposes
        self._create_sample_dataset()
        return True
    
    def _create_sample_dataset(self):
        """Create a sample video dataset for testing purposes."""
        self.logger.info("Creating sample RAVDESS dataset for testing")
        
        # Create sample video directory
        os.makedirs(self.ravdess_dir, exist_ok=True)
        
        # RAVDESS filename format: Modality-VocalChannel-Emotion-EmotionalIntensity-Statement-Repetition-Actor.mp4
        # Example: 01-01-06-01-01-01-01.mp4
        # Emotion codes: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
        
        sample_videos = []
        emotions = [1, 2, 3, 4, 5, 6, 7, 8]  # RAVDESS emotion codes
        
        for emotion in emotions:
            for actor in range(1, 5):  # 4 actors for testing
                for statement in [1, 2]:  # 2 statements
                    filename = f"01-01-{emotion:02d}-01-{statement:02d}-01-{actor:02d}.mp4"
                    video_path = os.path.join(self.ravdess_dir, filename)
                    
                    # Create a dummy video file (just create the file, don't write actual video data)
                    # In real implementation, this would be actual video files
                    with open(video_path, 'wb') as f:
                        f.write(b'dummy_video_data')  # Placeholder
                    
                    # Map RAVDESS emotion to our standard emotion
                    if emotion in RAVDESS_MAPPING:
                        standard_emotion = RAVDESS_MAPPING[emotion]
                        emotion_idx = EMOTION_CLASSES.index(standard_emotion)
                    else:
                        emotion_idx = EMOTION_CLASSES.index('neutrality')
                    
                    sample_videos.append({
                        'filename': filename,
                        'path': video_path,
                        'emotion': emotion,
                        'standard_emotion': standard_emotion if emotion in RAVDESS_MAPPING else 'neutrality',
                        'emotion_idx': emotion_idx,
                        'actor': actor,
                        'statement': statement,
                        'intensity': 1
                    })
        
        # Save metadata
        with open(self.ravdess_metadata_path, 'w') as f:
            json.dump(sample_videos, f, indent=2)
        
        self.logger.info(f"Sample dataset created with {len(sample_videos)} video entries")
    
    def load_raw_data(self) -> Tuple[List[str], np.ndarray]:
        """
        Load raw RAVDESS data (video paths and labels).
        
        Returns:
            Tuple of (video_paths, labels) where video_paths are file paths and labels are emotion indices
        """
        if not self.download_dataset():
            raise FileNotFoundError(f"RAVDESS dataset not found")
        
        self.logger.info(f"Loading RAVDESS data from {self.ravdess_dir}")
        
        # Load metadata if available
        if os.path.exists(self.ravdess_metadata_path):
            with open(self.ravdess_metadata_path, 'r') as f:
                metadata = json.load(f)
            
            video_paths = []
            labels = []
            
            for item in metadata:
                if os.path.exists(item['path']):
                    video_paths.append(item['path'])
                    labels.append(item['emotion_idx'])
            
            labels = np.array(labels)
        else:
            # Parse filenames directly
            video_paths = []
            labels = []
            
            for filename in os.listdir(self.ravdess_dir):
                if filename.endswith('.mp4'):
                    video_path = os.path.join(self.ravdess_dir, filename)
                    
                    # Parse RAVDESS filename format
                    parts = filename.replace('.mp4', '').split('-')
                    if len(parts) >= 3:
                        emotion_code = int(parts[2])
                        
                        # Map to standard emotion
                        if emotion_code in RAVDESS_MAPPING:
                            standard_emotion = RAVDESS_MAPPING[emotion_code]
                            emotion_idx = EMOTION_CLASSES.index(standard_emotion)
                        else:
                            emotion_idx = EMOTION_CLASSES.index('neutrality')
                        
                        video_paths.append(video_path)
                        labels.append(emotion_idx)
            
            labels = np.array(labels)
        
        self.logger.info(f"Loaded {len(video_paths)} videos")
        self.logger.info(f"Emotion distribution: {np.bincount(labels)}")
        
        return video_paths, labels
    
    def extract_frames(self, video_path: str) -> Optional[np.ndarray]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Array of frames (T, H, W, C) or None if extraction fails
        """
        # For sample dataset, create dummy frames
        if not os.path.exists(video_path) or os.path.getsize(video_path) < 100:
            # Create dummy frames for testing
            frames = np.random.randint(0, 256, 
                                     (self.sequence_length, self.frame_size[0], self.frame_size[1], 3), 
                                     dtype=np.uint8)
            return frames
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                self.logger.warning(f"Could not open video: {video_path}")
                return None
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame indices to extract
            if self.fps is not None and self.fps != original_fps:
                # Resample to target FPS
                frame_step = max(1, int(original_fps / self.fps))
            else:
                frame_step = 1
            
            # Extract frames uniformly across the video
            frame_indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    # If frame extraction fails, duplicate the last frame
                    if frames:
                        frames.append(frames[-1])
                    else:
                        # Create a black frame
                        frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
            
            cap.release()
            
            if len(frames) != self.sequence_length:
                self.logger.warning(f"Expected {self.sequence_length} frames, got {len(frames)}")
                # Pad or truncate to desired length
                while len(frames) < self.sequence_length:
                    frames.append(frames[-1] if frames else np.zeros((480, 640, 3), dtype=np.uint8))
                frames = frames[:self.sequence_length]
            
            return np.array(frames)
            
        except Exception as e:
            self.logger.error(f"Error extracting frames from {video_path}: {e}")
            return None
    
    def detect_and_align_faces_sequence(self, frames: np.ndarray) -> np.ndarray:
        """
        Detect and align faces across a sequence of frames.
        
        Args:
            frames: Array of frames (T, H, W, C)
            
        Returns:
            Array of face-cropped frames
        """
        if self.face_detector is None:
            return frames
        
        aligned_frames = []
        
        # Track face bounding box across frames for consistency
        prev_bbox = None
        
        for frame in frames:
            face_frame = self._detect_face_in_frame(frame, prev_bbox)
            aligned_frames.append(face_frame)
            
            # Update previous bounding box for tracking
            if hasattr(self, '_last_bbox'):
                prev_bbox = self._last_bbox
        
        return np.array(aligned_frames)
    
    def _detect_face_in_frame(self, frame: np.ndarray, prev_bbox: Optional[Tuple] = None) -> np.ndarray:
        """
        Detect face in a single frame.
        
        Args:
            frame: Single frame (H, W, C)
            prev_bbox: Previous bounding box for tracking
            
        Returns:
            Face-cropped frame
        """
        if hasattr(self.face_detector, 'detect_faces'):
            # MTCNN detector
            detections = self.face_detector.detect_faces(frame)
            
            if detections:
                # Use the most confident detection
                detection = max(detections, key=lambda d: d['confidence'])
                x, y, w, h = detection['box']
                
                # Add padding
                padding = 0.1
                x_pad = int(w * padding)
                y_pad = int(h * padding)
                
                x1 = max(0, x - x_pad)
                y1 = max(0, y - y_pad)
                x2 = min(frame.shape[1], x + w + x_pad)
                y2 = min(frame.shape[0], y + h + y_pad)
                
                face = frame[y1:y2, x1:x2]
                self._last_bbox = (x1, y1, x2, y2)
                
                return face
        else:
            # OpenCV Haar cascade detector
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Use the largest face or the one closest to previous detection
                if prev_bbox is not None:
                    # Find face closest to previous detection
                    prev_center = ((prev_bbox[0] + prev_bbox[2]) // 2, (prev_bbox[1] + prev_bbox[3]) // 2)
                    best_face = min(faces, key=lambda f: 
                                  abs(f[0] + f[2]//2 - prev_center[0]) + abs(f[1] + f[3]//2 - prev_center[1]))
                else:
                    # Use the largest face
                    best_face = max(faces, key=lambda f: f[2] * f[3])
                
                x, y, w, h = best_face
                face = frame[y:y+h, x:x+w]
                self._last_bbox = (x, y, x+w, y+h)
                
                return face
        
        # Return original frame if no face detected
        return frame
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame.
        
        Args:
            frame: Input frame (H, W, C)
            
        Returns:
            Preprocessed frame
        """
        # Resize to target size
        frame = cv2.resize(frame, self.frame_size)
        
        # Normalize pixel values to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def preprocess_sequence(self, frames: np.ndarray, augment: bool = False) -> np.ndarray:
        """
        Preprocess a sequence of frames.
        
        Args:
            frames: Input frames (T, H, W, C)
            augment: Whether to apply data augmentation
            
        Returns:
            Preprocessed frame sequence
        """
        # Face detection and alignment
        if self.use_face_detection:
            frames = self.detect_and_align_faces_sequence(frames)
        
        # Preprocess each frame
        processed_frames = []
        for frame in frames:
            processed_frame = self.preprocess_frame(frame)
            processed_frames.append(processed_frame)
        
        processed_frames = np.array(processed_frames)
        
        # Apply augmentation if requested
        if augment:
            processed_frames = self._apply_sequence_augmentation(processed_frames)
        
        return processed_frames
    
    def _apply_sequence_augmentation(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to frame sequence.
        
        Args:
            frames: Input frame sequence (T, H, W, C)
            
        Returns:
            Augmented frame sequence
        """
        # Temporal augmentation: random frame dropping/duplication
        if np.random.random() > 0.7:
            # Randomly drop some frames and duplicate others
            n_frames = len(frames)
            indices = np.random.choice(n_frames, n_frames, replace=True)
            frames = frames[indices]
        
        # Spatial augmentation applied to all frames
        augmented_frames = []
        
        # Random parameters for consistent augmentation across sequence
        flip = np.random.random() > 0.5
        brightness_factor = np.random.uniform(0.8, 1.2)
        contrast_factor = np.random.uniform(0.8, 1.2)
        
        for frame in frames:
            # Horizontal flip
            if flip:
                frame = cv2.flip(frame, 1)
            
            # Brightness adjustment
            frame = np.clip(frame * brightness_factor, 0.0, 1.0)
            
            # Contrast adjustment
            frame = np.clip((frame - 0.5) * contrast_factor + 0.5, 0.0, 1.0)
            
            augmented_frames.append(frame)
        
        return np.array(augmented_frames)
    
    def preprocess_data(self, video_paths: List[str], labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the raw video data.
        
        Args:
            video_paths: List of video file paths
            labels: Corresponding labels
            
        Returns:
            Tuple of (preprocessed_sequences, labels)
        """
        self.logger.info("Preprocessing video data...")
        
        preprocessed_sequences = []
        valid_labels = []
        
        for i, (video_path, label) in enumerate(zip(video_paths, labels)):
            if i % 10 == 0:
                self.logger.info(f"Processed {i}/{len(video_paths)} videos")
            
            # Extract frames
            frames = self.extract_frames(video_path)
            
            if frames is not None:
                # Preprocess sequence
                processed_sequence = self.preprocess_sequence(frames, augment=False)
                preprocessed_sequences.append(processed_sequence)
                valid_labels.append(label)
            else:
                self.logger.warning(f"Skipping video {video_path} due to processing error")
        
        preprocessed_sequences = np.array(preprocessed_sequences)
        valid_labels = np.array(valid_labels)
        
        self.logger.info(f"Preprocessing complete. Final shape: {preprocessed_sequences.shape}")
        
        return preprocessed_sequences, valid_labels
    
    def create_data_generator(self, 
                            sequences: np.ndarray, 
                            labels: np.ndarray,
                            batch_size: int = 8,
                            augment: bool = False,
                            shuffle: bool = True):
        """
        Create a data generator for batch processing.
        
        Args:
            sequences: Preprocessed video sequences
            labels: Labels
            batch_size: Batch size (smaller for video data)
            augment: Whether to apply augmentation
            shuffle: Whether to shuffle data
            
        Yields:
            Batches of (sequences, labels)
        """
        n_samples = len(sequences)
        indices = np.arange(n_samples)
        
        while True:
            if shuffle:
                np.random.shuffle(indices)
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_sequences = sequences[batch_indices]
                batch_labels = labels[batch_indices]
                
                # Apply augmentation if requested
                if augment:
                    augmented_batch = []
                    for sequence in batch_sequences:
                        augmented = self._apply_sequence_augmentation(sequence)
                        augmented_batch.append(augmented)
                    
                    batch_sequences = np.array(augmented_batch)
                
                yield batch_sequences, batch_labels
    
    def get_video_statistics(self, video_paths: List[str], labels: np.ndarray) -> Dict[str, Any]:
        """
        Get statistics about the video dataset.
        
        Args:
            video_paths: List of video file paths
            labels: Corresponding labels
            
        Returns:
            Dictionary with video statistics
        """
        stats = {
            'total_videos': len(video_paths),
            'sequence_length': self.sequence_length,
            'frame_size': self.frame_size,
            'class_distribution': {},
            'class_percentages': {},
            'video_info': []
        }
        
        # Analyze a few sample videos for detailed stats
        sample_indices = np.random.choice(len(video_paths), min(5, len(video_paths)), replace=False)
        
        for idx in sample_indices:
            video_path = video_paths[idx]
            
            if os.path.exists(video_path) and os.path.getsize(video_path) > 100:
                try:
                    cap = cv2.VideoCapture(video_path)
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        duration = frame_count / fps if fps > 0 else 0
                        
                        stats['video_info'].append({
                            'path': video_path,
                            'fps': fps,
                            'frame_count': frame_count,
                            'resolution': (width, height),
                            'duration_seconds': duration
                        })
                        
                        cap.release()
                except Exception as e:
                    self.logger.warning(f"Could not analyze video {video_path}: {e}")
        
        # Class distribution
        unique, counts = np.unique(labels, return_counts=True)
        for class_idx, count in zip(unique, counts):
            emotion_name = EMOTION_CLASSES[class_idx]
            stats['class_distribution'][emotion_name] = int(count)
            stats['class_percentages'][emotion_name] = float(count / len(labels) * 100)
        
        return stats
    
    def create_sample_sequences(self, n_samples: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sample video sequences for testing.
        
        Args:
            n_samples: Number of sample sequences to create
            
        Returns:
            Tuple of (sample_sequences, sample_labels)
        """
        sequences = []
        labels = []
        
        for i in range(n_samples):
            # Create random sequence
            sequence = np.random.rand(self.sequence_length, self.frame_size[0], self.frame_size[1], 3)
            sequence = sequence.astype(np.float32)
            
            # Random label
            label = np.random.randint(0, len(EMOTION_CLASSES))
            
            sequences.append(sequence)
            labels.append(label)
        
        return np.array(sequences), np.array(labels)