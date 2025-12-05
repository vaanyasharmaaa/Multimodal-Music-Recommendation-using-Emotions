"""
Text data loader for emotion text datasets with preprocessing and augmentation.
"""

import os
import numpy as np
import pandas as pd
import re
import json
import requests
from typing import Tuple, List, Optional, Dict, Any
import logging
from collections import Counter

from .data_loaders import BaseDataLoader
from .constants import TEXT_EMOTION_MAPPING, EMOTION_CLASSES


class TextDataLoader(BaseDataLoader):
    """
    Data loader for text-based emotion recognition.
    
    Handles dataset download, text preprocessing, tokenization with DistilBERT,
    and text augmentation techniques for emotion classification tasks.
    """
    
    def __init__(self, 
                 data_dir: str = "data/raw",
                 processed_dir: str = "data/processed",
                 cache_dir: str = "data/cache",
                 max_length: int = 512,
                 tokenizer_name: str = "distilbert-base-uncased",
                 random_state: int = 42):
        """
        Initialize the text data loader.
        
        Args:
            data_dir: Directory containing raw datasets
            processed_dir: Directory for processed datasets
            cache_dir: Directory for cached preprocessed data
            max_length: Maximum sequence length for tokenization
            tokenizer_name: Name of the tokenizer to use
            random_state: Random seed for reproducibility
        """
        super().__init__(data_dir, processed_dir, cache_dir, random_state)
        
        self.max_length = max_length
        self.tokenizer_name = tokenizer_name
        self.dataset_name = "emotion_text"
        
        # Initialize tokenizer
        self.tokenizer = None
        self._initialize_tokenizer()
        
        # Dataset paths
        self.emotion_dataset_path = os.path.join(self.data_dir, "emotion_dataset.json")
        
    def _initialize_tokenizer(self):
        """Initialize DistilBERT tokenizer."""
        try:
            from transformers import DistilBertTokenizer
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.tokenizer_name)
            self.logger.info(f"DistilBERT tokenizer initialized: {self.tokenizer_name}")
        except ImportError:
            self.logger.warning("Transformers library not available, using basic tokenization")
            self.tokenizer = None
        except Exception as e:
            self.logger.warning(f"Could not load tokenizer: {e}, using basic tokenization")
            self.tokenizer = None
    
    def download_dataset(self) -> bool:
        """
        Download emotion text dataset.
        
        Note: This method creates a sample dataset for testing since
        real datasets may require specific access or APIs.
        
        Returns:
            bool: True if dataset is available, False otherwise
        """
        if os.path.exists(self.emotion_dataset_path):
            self.logger.info("Emotion text dataset already exists")
            return True
        
        # Check if we have a sample dataset for testing
        sample_path = os.path.join(self.data_dir, "emotion_sample.json")
        if os.path.exists(sample_path):
            self.emotion_dataset_path = sample_path
            self.logger.info("Using sample emotion text dataset")
            return True
        
        # Create a sample dataset for testing purposes
        self._create_sample_dataset()
        return True
    
    def _create_sample_dataset(self):
        """Create a sample text emotion dataset for testing purposes."""
        self.logger.info("Creating sample emotion text dataset for testing")
        
        # Sample texts for each emotion
        sample_texts = {
            'anger': [
                "I am so frustrated with this situation!",
                "This makes me really angry and upset.",
                "I can't believe how infuriating this is.",
                "This is absolutely outrageous!",
                "I'm furious about what happened.",
            ],
            'disgust': [
                "That is absolutely disgusting and revolting.",
                "I find this completely repulsive.",
                "This is so gross and nauseating.",
                "I'm disgusted by this behavior.",
                "That's utterly repugnant.",
            ],
            'fear': [
                "I'm really scared about what might happen.",
                "This situation terrifies me completely.",
                "I'm afraid of the consequences.",
                "This makes me feel very anxious and worried.",
                "I'm frightened by this possibility.",
            ],
            'happiness': [
                "I'm so happy and excited about this!",
                "This brings me so much joy and delight.",
                "I feel absolutely wonderful today.",
                "This is amazing and makes me smile.",
                "I'm thrilled and overjoyed!",
            ],
            'neutrality': [
                "This is a normal everyday occurrence.",
                "The weather is okay today.",
                "I went to the store and bought groceries.",
                "The meeting was scheduled for 3 PM.",
                "This is a factual statement.",
            ],
            'sadness': [
                "I feel so sad and depressed about this.",
                "This makes me feel really down and blue.",
                "I'm heartbroken by what happened.",
                "This situation brings tears to my eyes.",
                "I feel melancholy and sorrowful.",
            ]
        }
        
        # Generate dataset with multiple samples per emotion
        dataset = []
        for emotion, texts in sample_texts.items():
            emotion_idx = EMOTION_CLASSES.index(emotion)
            
            # Add original texts
            for text in texts:
                dataset.append({
                    'text': text,
                    'emotion': emotion,
                    'emotion_idx': emotion_idx
                })
            
            # Add some variations
            for i, text in enumerate(texts[:3]):  # Take first 3 for variations
                # Simple variations
                variations = [
                    text.replace("I'm", "I am"),
                    text.replace(".", "!"),
                    text.capitalize(),
                ]
                
                for variation in variations:
                    if variation != text:  # Avoid duplicates
                        dataset.append({
                            'text': variation,
                            'emotion': emotion,
                            'emotion_idx': emotion_idx
                        })
        
        # Save dataset
        sample_path = os.path.join(self.data_dir, "emotion_sample.json")
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        self.emotion_dataset_path = sample_path
        self.logger.info(f"Sample dataset created with {len(dataset)} samples")
    
    def load_raw_data(self) -> Tuple[List[str], np.ndarray]:
        """
        Load raw text data from JSON file.
        
        Returns:
            Tuple of (texts, labels) where texts are strings and labels are emotion indices
        """
        if not os.path.exists(self.emotion_dataset_path):
            if not self.download_dataset():
                raise FileNotFoundError(f"Emotion dataset not found at {self.emotion_dataset_path}")
        
        self.logger.info(f"Loading emotion text data from {self.emotion_dataset_path}")
        
        # Load JSON data
        with open(self.emotion_dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        labels = []
        
        for item in data:
            text = item['text']
            emotion = item.get('emotion', item.get('label', 'neutrality'))
            
            # Clean and preprocess text
            cleaned_text = self._basic_text_cleaning(text)
            texts.append(cleaned_text)
            
            # Map emotion to standard index
            if emotion in EMOTION_CLASSES:
                emotion_idx = EMOTION_CLASSES.index(emotion)
            elif emotion.lower() in TEXT_EMOTION_MAPPING:
                standard_emotion = TEXT_EMOTION_MAPPING[emotion.lower()]
                emotion_idx = EMOTION_CLASSES.index(standard_emotion)
            else:
                # Default to neutrality for unknown emotions
                emotion_idx = EMOTION_CLASSES.index('neutrality')
            
            labels.append(emotion_idx)
        
        labels = np.array(labels)
        
        self.logger.info(f"Loaded {len(texts)} text samples")
        self.logger.info(f"Emotion distribution: {np.bincount(labels)}")
        
        return texts, labels
    
    def _basic_text_cleaning(self, text: str) -> str:
        """
        Apply basic text cleaning.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation that might be important for emotion
        text = re.sub(r'[^\w\s.,!?;:\'-]', ' ', text)
        
        # Remove extra whitespace again after character removal
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize_and_encode(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Tokenize and encode texts using DistilBERT tokenizer.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with input_ids, attention_mask, and token_type_ids
        """
        if self.tokenizer is None:
            # Fallback to basic tokenization
            return self._basic_tokenization(texts)
        
        self.logger.info(f"Tokenizing {len(texts)} texts with DistilBERT tokenizer")
        
        # Tokenize all texts
        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def _basic_tokenization(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Fallback basic tokenization when transformers is not available.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with basic token representations
        """
        self.logger.info("Using basic tokenization (transformers not available)")
        
        # Create a simple vocabulary from all texts
        all_words = []
        for text in texts:
            words = text.lower().split()
            all_words.extend(words)
        
        # Create vocabulary (most common words)
        word_counts = Counter(all_words)
        vocab = ['<PAD>', '<UNK>'] + [word for word, _ in word_counts.most_common(5000)]
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        
        # Tokenize texts
        input_ids = []
        attention_masks = []
        
        for text in texts:
            words = text.lower().split()[:self.max_length-2]  # Leave space for special tokens
            
            # Convert words to indices
            token_ids = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words]
            
            # Pad or truncate to max_length
            if len(token_ids) < self.max_length:
                attention_mask = [1] * len(token_ids) + [0] * (self.max_length - len(token_ids))
                token_ids = token_ids + [word_to_idx['<PAD>']] * (self.max_length - len(token_ids))
            else:
                token_ids = token_ids[:self.max_length]
                attention_mask = [1] * self.max_length
            
            input_ids.append(token_ids)
            attention_masks.append(attention_mask)
        
        return {
            'input_ids': np.array(input_ids, dtype=np.int64),
            'attention_mask': np.array(attention_masks, dtype=np.int64)
        }
    
    def preprocess_data(self, X: List[str], y: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Preprocess the raw text data.
        
        Args:
            X: Raw text strings
            y: Raw labels
            
        Returns:
            Tuple of (tokenized_data, labels)
        """
        self.logger.info("Preprocessing text data...")
        
        # Apply additional text cleaning
        cleaned_texts = []
        for text in X:
            cleaned = self._advanced_text_cleaning(text)
            cleaned_texts.append(cleaned)
        
        # Tokenize and encode
        tokenized_data = self.tokenize_and_encode(cleaned_texts)
        
        self.logger.info(f"Preprocessing complete. Input shape: {tokenized_data['input_ids'].shape}")
        
        return tokenized_data, y
    
    def _advanced_text_cleaning(self, text: str) -> str:
        """
        Apply advanced text cleaning and preprocessing.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned and preprocessed text
        """
        # Convert to lowercase first
        text = text.lower()
        
        # Remove URLs (before other cleaning)
        text = re.sub(r'https?://[^\s]+|www\.[^\s]+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'[^\s]+@[^\s]+', '', text)
        
        # Expand contractions
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{2,}', '.', text)
        
        # Remove special characters but keep punctuation that might be important for emotion
        text = re.sub(r'[^\w\s.,!?;:\'-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def apply_text_augmentation(self, texts: List[str], labels: np.ndarray, 
                              augmentation_factor: float = 0.2) -> Tuple[List[str], np.ndarray]:
        """
        Apply text augmentation techniques.
        
        Args:
            texts: Original texts
            labels: Corresponding labels
            augmentation_factor: Fraction of data to augment
            
        Returns:
            Tuple of (augmented_texts, augmented_labels)
        """
        self.logger.info(f"Applying text augmentation with factor {augmentation_factor}")
        
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()
        
        n_augment = int(len(texts) * augmentation_factor)
        indices_to_augment = np.random.choice(len(texts), n_augment, replace=False)
        
        for idx in indices_to_augment:
            original_text = texts[idx]
            label = labels[idx]
            
            # Apply random augmentation technique
            augmentation_type = np.random.choice(['synonym', 'insertion', 'deletion', 'swap'])
            
            if augmentation_type == 'synonym':
                augmented_text = self._synonym_replacement(original_text)
            elif augmentation_type == 'insertion':
                augmented_text = self._random_insertion(original_text)
            elif augmentation_type == 'deletion':
                augmented_text = self._random_deletion(original_text)
            else:  # swap
                augmented_text = self._random_swap(original_text)
            
            if augmented_text != original_text:  # Only add if actually changed
                augmented_texts.append(augmented_text)
                augmented_labels = np.append(augmented_labels, label)
        
        self.logger.info(f"Augmentation complete. Original: {len(texts)}, Augmented: {len(augmented_texts)}")
        
        return augmented_texts, augmented_labels
    
    def _synonym_replacement(self, text: str, n_replacements: int = 1) -> str:
        """Simple synonym replacement using basic word substitutions."""
        # Basic synonym dictionary for common emotion words
        synonyms = {
            'happy': ['joyful', 'glad', 'cheerful', 'pleased'],
            'sad': ['unhappy', 'sorrowful', 'melancholy', 'dejected'],
            'angry': ['furious', 'mad', 'irritated', 'enraged'],
            'scared': ['afraid', 'frightened', 'terrified', 'worried'],
            'good': ['great', 'excellent', 'wonderful', 'fantastic'],
            'bad': ['terrible', 'awful', 'horrible', 'dreadful'],
            'big': ['large', 'huge', 'enormous', 'massive'],
            'small': ['tiny', 'little', 'miniature', 'petite']
        }
        
        words = text.split()
        for _ in range(n_replacements):
            if not words:
                break
                
            # Find words that have synonyms
            replaceable_words = [i for i, word in enumerate(words) if word.lower() in synonyms]
            
            if replaceable_words:
                idx = np.random.choice(replaceable_words)
                original_word = words[idx].lower()
                synonym = np.random.choice(synonyms[original_word])
                
                # Preserve original case
                if words[idx].isupper():
                    words[idx] = synonym.upper()
                elif words[idx].istitle():
                    words[idx] = synonym.capitalize()
                else:
                    words[idx] = synonym
        
        return ' '.join(words)
    
    def _random_insertion(self, text: str, n_insertions: int = 1) -> str:
        """Insert random words from the text at random positions."""
        words = text.split()
        if len(words) < 2:
            return text
        
        for _ in range(n_insertions):
            # Choose a random word from the text
            random_word = np.random.choice(words)
            # Choose a random position to insert
            random_idx = np.random.randint(0, len(words) + 1)
            words.insert(random_idx, random_word)
        
        return ' '.join(words)
    
    def _random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words with probability p."""
        words = text.split()
        if len(words) <= 1:
            return text
        
        new_words = []
        for word in words:
            if np.random.random() > p:
                new_words.append(word)
        
        # Ensure at least one word remains
        if not new_words:
            new_words = [np.random.choice(words)]
        
        return ' '.join(new_words)
    
    def _random_swap(self, text: str, n_swaps: int = 1) -> str:
        """Randomly swap positions of words."""
        words = text.split()
        if len(words) < 2:
            return text
        
        for _ in range(n_swaps):
            idx1, idx2 = np.random.choice(len(words), 2, replace=False)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def create_data_generator(self, 
                            tokenized_data: Dict[str, np.ndarray], 
                            labels: np.ndarray,
                            batch_size: int = 32,
                            shuffle: bool = True):
        """
        Create a data generator for batch processing.
        
        Args:
            tokenized_data: Dictionary with tokenized inputs
            labels: Labels array
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Yields:
            Batches of (tokenized_inputs, labels)
        """
        n_samples = len(labels)
        indices = np.arange(n_samples)
        
        while True:
            if shuffle:
                np.random.shuffle(indices)
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_data = {}
                for key, values in tokenized_data.items():
                    batch_data[key] = values[batch_indices]
                
                batch_labels = labels[batch_indices]
                
                yield batch_data, batch_labels
    
    def get_text_statistics(self, texts: List[str], labels: np.ndarray) -> Dict[str, Any]:
        """
        Get statistics about the text dataset.
        
        Args:
            texts: List of text strings
            labels: Corresponding labels
            
        Returns:
            Dictionary with text statistics
        """
        # Text length statistics
        text_lengths = [len(text.split()) for text in texts]
        
        # Character statistics
        char_lengths = [len(text) for text in texts]
        
        # Vocabulary statistics
        all_words = []
        for text in texts:
            all_words.extend(text.lower().split())
        
        vocab_size = len(set(all_words))
        word_counts = Counter(all_words)
        
        stats = {
            'total_samples': len(texts),
            'avg_text_length_words': np.mean(text_lengths),
            'max_text_length_words': np.max(text_lengths),
            'min_text_length_words': np.min(text_lengths),
            'avg_text_length_chars': np.mean(char_lengths),
            'vocabulary_size': vocab_size,
            'most_common_words': word_counts.most_common(10),
            'class_distribution': {},
            'class_percentages': {}
        }
        
        # Class distribution
        unique, counts = np.unique(labels, return_counts=True)
        for class_idx, count in zip(unique, counts):
            emotion_name = EMOTION_CLASSES[class_idx]
            stats['class_distribution'][emotion_name] = int(count)
            stats['class_percentages'][emotion_name] = float(count / len(labels) * 100)
        
        return stats