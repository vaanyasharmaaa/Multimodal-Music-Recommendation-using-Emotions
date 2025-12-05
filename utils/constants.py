"""
Constants and mappings for the multi-modal emotion detection system.
"""

# Emotion mapping dictionary - maps various emotion labels to standardized 6-class system
EMOTION_MAPPING = {
    'anger': 0,
    'angry': 0,
    'disgust': 1,
    'disgusted': 1,
    'fear': 2,
    'fearful': 2,
    'afraid': 2,
    'happiness': 3,
    'happy': 3,
    'joy': 3,
    'joyful': 3,
    'neutrality': 4,
    'neutral': 4,
    'calm': 4,
    'sadness': 5,
    'sad': 5,
    'sorrow': 5
}

# Reverse mapping for converting indices back to emotion names
INDEX_TO_EMOTION = {
    0: 'anger',
    1: 'disgust', 
    2: 'fear',
    3: 'happiness',
    4: 'neutrality',
    5: 'sadness'
}

# List of target emotion classes
EMOTION_CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'neutrality', 'sadness']

# Number of emotion classes
NUM_CLASSES = 6

# Dataset-specific mappings for different datasets
FER2013_MAPPING = {
    0: 'anger',      # angry
    1: 'disgust',    # disgust  
    2: 'fear',       # fear
    3: 'happiness',  # happy
    4: 'sadness',    # sad
    5: 'neutrality', # surprise -> mapped to neutrality
    6: 'neutrality'  # neutral
}

RAVDESS_MAPPING = {
    1: 'neutrality', # neutral
    2: 'neutrality', # calm
    3: 'happiness',  # happy
    4: 'sadness',    # sad
    5: 'anger',      # angry
    6: 'fear',       # fearful
    7: 'disgust',    # disgust
    8: 'neutrality'  # surprised -> mapped to neutrality
}

# Text dataset emotion mappings (for various text emotion datasets)
TEXT_EMOTION_MAPPING = {
    'joy': 'happiness',
    'surprise': 'neutrality',
    'love': 'happiness',
    'optimism': 'happiness',
    'pessimism': 'sadness',
    'trust': 'neutrality'
}