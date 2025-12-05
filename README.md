# Multi-Modal Emotion Detection System

A comprehensive emotion detection system that classifies emotions from images, text, and video using deep learning. The system includes individual models for each modality as well as a multi-modal fusion model that combines all three input types.

## Features

- **Image Emotion Detection**: CNN-based model using ResNet-50 for facial emotion recognition
- **Text Emotion Detection**: Transformer-based model using DistilBERT for text emotion classification  
- **Video Emotion Detection**: 3D CNN + LSTM hybrid model for spatio-temporal emotion recognition
- **Multi-Modal Fusion**: Attention-based fusion model combining all three modalities
- **Comprehensive Evaluation**: Cross-validation, performance metrics, and model comparison tools
- **Real-time Demo**: Live emotion detection from camera feed

## Emotion Classes

The system classifies emotions into 6 categories:
- Anger
- Disgust  
- Fear
- Happiness
- Neutrality
- Sadness

## Project Structure

```
├── config/                 # Configuration files
│   ├── model_config.yaml   # Model hyperparameters
│   ├── dataset_config.yaml # Dataset paths and preprocessing
│   └── config.py          # Configuration management
├── data/                   # Dataset storage
│   ├── raw/               # Original datasets
│   ├── processed/         # Preprocessed data
│   ├── splits/            # Train/val/test splits
│   └── cache/             # Cached data for faster loading
├── models/                 # Trained models
│   ├── image/             # Image emotion models
│   ├── text/              # Text emotion models
│   ├── video/             # Video emotion models
│   ├── multimodal/        # Multi-modal fusion models
│   └── checkpoints/       # Training checkpoints
├── notebooks/              # Jupyter notebooks
│   ├── image_emotion_detection.ipynb
│   ├── text_emotion_detection.ipynb
│   ├── video_emotion_detection.ipynb
│   ├── multimodal_fusion.ipynb
│   ├── evaluation_comparison.ipynb
│   └── realtime_demo.ipynb
├── utils/                  # Utility modules
└── requirements.txt        # Python dependencies
```

## Quick Start

1. **Setup Environment**
   ```bash
   python setup.py
   ```

2. **Start Training**
   - Open any notebook in the `notebooks/` directory
   - Follow the step-by-step instructions in each notebook
   - Models will be automatically saved to the `models/` directory

3. **Customize Configuration**
   - Modify `config/model_config.yaml` for model hyperparameters
   - Modify `config/dataset_config.yaml` for dataset settings

## Datasets

The system uses publicly available datasets:

- **FER-2013**: Facial expression recognition dataset for image emotion detection
- **Emotion Dataset for NLP**: Hugging Face dataset for text emotion classification
- **RAVDESS**: Ryerson Audio-Visual Database for video emotion recognition

Datasets are automatically downloaded when running the notebooks.

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- PyTorch 2.0+
- Transformers 4.30+
- OpenCV 4.8+
- See `requirements.txt` for complete list

## Usage

### Training Individual Models

```python
# Example: Train image emotion detection model
from utils.config import config
from notebooks import image_emotion_detection

# Configuration is automatically loaded
model = image_emotion_detection.train_model()
```

### Multi-Modal Inference

```python
# Example: Predict emotion from multiple modalities
from utils.multimodal_fusion import MultiModalPredictor

predictor = MultiModalPredictor()
emotion = predictor.predict(
    image_path="path/to/image.jpg",
    text="I am feeling great today!",
    video_path="path/to/video.mp4"
)
```

## Performance

The system achieves competitive performance on standard emotion recognition benchmarks:

- Image Model: ~70% accuracy on FER-2013
- Text Model: ~85% accuracy on Emotion Dataset
- Video Model: ~75% accuracy on RAVDESS  
- Multi-Modal: ~80% accuracy (combined evaluation)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{multimodal_emotion_detection,
  title={Multi-Modal Emotion Detection System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/multimodal-emotion-detection}
}
```