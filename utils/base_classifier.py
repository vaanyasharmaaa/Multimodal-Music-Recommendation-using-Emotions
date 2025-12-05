"""
Base emotion classifier class providing common functionality for all emotion detection models.
"""

import os
import json
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, SparseCategoricalAccuracy, Precision, Recall

from .constants import NUM_CLASSES, EMOTION_CLASSES, INDEX_TO_EMOTION


class EmotionClassifier(ABC):
    """
    Abstract base class for emotion classification models.
    
    Provides common functionality for training, evaluation, and prediction
    across different modalities (image, text, video).
    """
    
    def __init__(self, 
                 num_classes: int = NUM_CLASSES,
                 model_name: str = "emotion_classifier",
                 **kwargs):
        """
        Initialize the emotion classifier.
        
        Args:
            num_classes: Number of emotion classes to predict
            model_name: Name identifier for the model
            **kwargs: Additional model-specific parameters
        """
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = None
        self.history = None
        self.emotion_classes = EMOTION_CLASSES[:num_classes]
        
        # Training configuration
        self.config = {
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001,
            'patience': 10,
            'min_delta': 0.001,
            'factor': 0.5,
            'min_lr': 1e-7,
            **kwargs
        }
    
    @abstractmethod
    def build_model(self) -> tf.keras.Model:
        """
        Build the model architecture.
        
        Returns:
            Compiled Keras model
        """
        pass
    
    def compile_model(self, 
                     optimizer: str = 'adam',
                     loss: str = 'categorical_crossentropy',
                     metrics: List[str] = None,
                     **optimizer_kwargs) -> None:
        """
        Compile the model with specified optimizer, loss, and metrics.
        
        Args:
            optimizer: Optimizer name or instance
            loss: Loss function name or instance
            metrics: List of metrics to track
            **optimizer_kwargs: Additional optimizer parameters
        """
        if self.model is None:
            self.model = self.build_model()
        
        # Configure optimizer
        if isinstance(optimizer, str):
            if optimizer.lower() == 'adam':
                opt = Adam(learning_rate=self.config['learning_rate'], **optimizer_kwargs)
            elif optimizer.lower() == 'adamw':
                opt = AdamW(learning_rate=self.config['learning_rate'], **optimizer_kwargs)
            else:
                opt = optimizer
        else:
            opt = optimizer
        
        # Configure loss function
        if isinstance(loss, str):
            if loss == 'categorical_crossentropy':
                loss_fn = CategoricalCrossentropy(label_smoothing=0.1)
            elif loss == 'sparse_categorical_crossentropy':
                loss_fn = SparseCategoricalCrossentropy()
            else:
                loss_fn = loss
        else:
            loss_fn = loss
        
        # Configure metrics
        if metrics is None:
            if 'categorical' in str(loss_fn).lower():
                metrics = [CategoricalAccuracy(name='accuracy'), 
                          Precision(name='precision'), 
                          Recall(name='recall')]
            else:
                metrics = [SparseCategoricalAccuracy(name='accuracy'),
                          Precision(name='precision'),
                          Recall(name='recall')]
        
        self.model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)
    
    def get_callbacks(self, 
                     checkpoint_path: str = None,
                     monitor: str = 'val_loss',
                     mode: str = 'min') -> List[tf.keras.callbacks.Callback]:
        """
        Get training callbacks for monitoring and early stopping.
        
        Args:
            checkpoint_path: Path to save model checkpoints
            monitor: Metric to monitor for callbacks
            mode: Whether to minimize or maximize the monitored metric
            
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor=monitor,
            patience=self.config['patience'],
            min_delta=self.config['min_delta'],
            mode=mode,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        lr_scheduler = ReduceLROnPlateau(
            monitor=monitor,
            factor=self.config['factor'],
            patience=self.config['patience'] // 2,
            min_lr=self.config['min_lr'],
            mode=mode,
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # Model checkpoint
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            checkpoint = ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=monitor,
                save_best_only=True,
                save_weights_only=False,
                mode=mode,
                verbose=1
            )
            callbacks.append(checkpoint)
        
        return callbacks
    
    def train(self, 
              train_data,
              validation_data=None,
              epochs: int = None,
              batch_size: int = None,
              checkpoint_path: str = None,
              **kwargs) -> tf.keras.callbacks.History:
        """
        Train the emotion classifier model.
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset
            epochs: Number of training epochs
            batch_size: Batch size for training
            checkpoint_path: Path to save model checkpoints
            **kwargs: Additional training parameters
            
        Returns:
            Training history object
        """
        if self.model is None:
            raise ValueError("Model must be built and compiled before training")
        
        # Use config values if not provided
        epochs = epochs or self.config['epochs']
        batch_size = batch_size or self.config['batch_size']
        
        # Get callbacks
        callbacks = self.get_callbacks(checkpoint_path)
        
        # Train the model
        self.history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            **kwargs
        )
        
        return self.history
    
    def evaluate(self, test_data, batch_size: int = None) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test dataset
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        batch_size = batch_size or self.config['batch_size']
        
        # Evaluate model
        results = self.model.evaluate(test_data, batch_size=batch_size, verbose=1)
        
        # Create results dictionary
        metric_names = self.model.metrics_names
        results_dict = dict(zip(metric_names, results))
        
        return results_dict
    
    def predict(self, 
                input_data,
                batch_size: int = None,
                return_probabilities: bool = True) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            input_data: Input data for prediction
            batch_size: Batch size for prediction
            return_probabilities: Whether to return probabilities or class indices
            
        Returns:
            Predictions as numpy array
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        batch_size = batch_size or self.config['batch_size']
        
        # Make predictions
        predictions = self.model.predict(input_data, batch_size=batch_size, verbose=0)
        
        if return_probabilities:
            return predictions
        else:
            return np.argmax(predictions, axis=1)
    
    def predict_emotion(self, input_data, batch_size: int = None) -> List[Dict[str, Any]]:
        """
        Predict emotions with class names and confidence scores.
        
        Args:
            input_data: Input data for prediction
            batch_size: Batch size for prediction
            
        Returns:
            List of dictionaries containing emotion predictions
        """
        probabilities = self.predict(input_data, batch_size, return_probabilities=True)
        
        results = []
        for prob in probabilities:
            predicted_class = np.argmax(prob)
            confidence = float(prob[predicted_class])
            emotion = INDEX_TO_EMOTION[predicted_class]
            
            result = {
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': {
                    self.emotion_classes[i]: float(prob[i]) 
                    for i in range(len(self.emotion_classes))
                }
            }
            results.append(result)
        
        return results
    
    def save_model(self, 
                   filepath: str,
                   save_format: str = 'tf',
                   include_config: bool = True) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
            save_format: Format to save ('tf', 'h5')
            include_config: Whether to save model configuration
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        self.model.save(filepath, save_format=save_format)
        
        # Save configuration
        if include_config:
            config_path = filepath.replace('.h5', '_config.json').replace('.tf', '_config.json')
            with open(config_path, 'w') as f:
                json.dump({
                    'model_name': self.model_name,
                    'num_classes': self.num_classes,
                    'emotion_classes': self.emotion_classes,
                    'config': self.config
                }, f, indent=2)
    
    def load_model(self, 
                   filepath: str,
                   load_config: bool = True) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            load_config: Whether to load model configuration
        """
        # Load model
        self.model = tf.keras.models.load_model(filepath)
        
        # Load configuration
        if load_config:
            config_path = filepath.replace('.h5', '_config.json').replace('.tf', '_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                    self.model_name = saved_config.get('model_name', self.model_name)
                    self.num_classes = saved_config.get('num_classes', self.num_classes)
                    self.emotion_classes = saved_config.get('emotion_classes', self.emotion_classes)
                    self.config.update(saved_config.get('config', {}))
    
    def get_model_summary(self) -> str:
        """
        Get a string representation of the model architecture.
        
        Returns:
            Model summary as string
        """
        if self.model is None:
            return "Model not built yet"
        
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.model.summary()
        sys.stdout = old_stdout
        
        return buffer.getvalue()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current model configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'emotion_classes': self.emotion_classes,
            'config': self.config.copy()
        }