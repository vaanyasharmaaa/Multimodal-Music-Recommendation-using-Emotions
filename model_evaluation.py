import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import pandas as pd

# Load model and setup
model = load_model('model.h5')
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# Data generator for test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    "images/test/",
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=False
)

# Get predictions
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score (Macro): {f1_macro:.4f}")
print(f"F1-Score (Weighted): {f1_weighted:.4f}")

# Classification report
report = classification_report(y_true, y_pred, target_names=emotion_labels, output_dict=True)
print("\nClassification Report:")
print(pd.DataFrame(report).transpose().round(4))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plotting
plt.figure(figsize=(15, 10))

# Confusion Matrix
plt.subplot(2, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.title('Confusion Matrix')

# Normalized Confusion Matrix
plt.subplot(2, 3, 2)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.title('Normalized Confusion Matrix')

# Per-class metrics
plt.subplot(2, 3, 3)
metrics_df = pd.DataFrame({
    'Precision': [report[emotion]['precision'] for emotion in emotion_labels],
    'Recall': [report[emotion]['recall'] for emotion in emotion_labels],
    'F1-Score': [report[emotion]['f1-score'] for emotion in emotion_labels]
}, index=emotion_labels)
metrics_df.plot(kind='bar', ax=plt.gca())
plt.title('Per-Class Metrics')
plt.xticks(rotation=45)

# Class distribution
plt.subplot(2, 3, 4)
unique, counts = np.unique(y_true, return_counts=True)
plt.bar([emotion_labels[i] for i in unique], counts)
plt.title('Test Set Distribution')
plt.xticks(rotation=45)

# Prediction confidence
plt.subplot(2, 3, 5)
max_probs = np.max(predictions, axis=1)
plt.hist(max_probs, bins=30, alpha=0.7)
plt.title('Prediction Confidence')
plt.xlabel('Max Probability')

# F1 scores by class
plt.subplot(2, 3, 6)
f1_per_class = f1_score(y_true, y_pred, average=None)
plt.bar(emotion_labels, f1_per_class)
plt.title('F1-Score by Class')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Summary
best_class = emotion_labels[np.argmax(f1_per_class)]
worst_class = emotion_labels[np.argmin(f1_per_class)]
print(f"\nBest performing: {best_class} (F1: {max(f1_per_class):.4f})")
print(f"Worst performing: {worst_class} (F1: {min(f1_per_class):.4f})")
print(f"Average confidence: {np.mean(max_probs):.4f}")