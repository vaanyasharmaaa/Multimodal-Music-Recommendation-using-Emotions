import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

# Importing Deep Learning Libraries
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Input,Dropout,GlobalAveragePooling2D,Flatten,Conv2D,BatchNormalization,Activation,MaxPooling2D
from keras.models import Model,Sequential
from keras.optimizers import Adam,SGD,RMSprop

# Configuration
picture_size = 48
folder_path = "./images/"  # Updated path for local dataset
batch_size = 128

# Check if dataset exists
if not os.path.exists(folder_path):
    print(f"Dataset not found at {folder_path}")
    print("Please ensure the dataset is in the correct location")
    exit()

# Data generators
datagen_train = ImageDataGenerator()
datagen_val = ImageDataGenerator()

# Define classes excluding 'disgust'
emotion_classes = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Training data generator
train_set = datagen_train.flow_from_directory(folder_path+"train",
                                              target_size = (picture_size,picture_size),
                                              color_mode = "grayscale",
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              classes=emotion_classes,
                                              shuffle=True)

# Test data generator for validation
test_set = datagen_val.flow_from_directory(folder_path+"test",
                                           target_size = (picture_size,picture_size),
                                           color_mode = "grayscale",
                                           batch_size=batch_size,
                                           class_mode='categorical',
                                           classes=emotion_classes,
                                           shuffle=False)

print(f"Found {train_set.n} training images and {test_set.n} validation images")

# Model Building
no_of_classes = 6  # Changed from 7 to 6

model = Sequential()

#1st CNN layer
model.add(Conv2D(64,(3,3),padding = 'same',input_shape = (48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

#2nd CNN layer
model.add(Conv2D(128,(5,5),padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout (0.25))

#3rd CNN layer
model.add(Conv2D(512,(3,3),padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout (0.25))

#4th CNN layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

#Fully connected 1st layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Fully connected layer 2nd layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(no_of_classes, activation='softmax'))

# Compile model
opt = Adam(lr = 0.0001)
model.compile(optimizer=opt,loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training setup
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import time

# Custom progress callback
class ProgressCallback(Callback):
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.start_time = None
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"\n🚀 Starting training for {self.total_epochs} epochs...")
        
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n📊 Epoch {epoch + 1}/{self.total_epochs}")
        
    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start_time
        progress = (epoch + 1) / self.total_epochs * 100
        eta = elapsed / (epoch + 1) * (self.total_epochs - epoch - 1)
        
        print(f"✅ Progress: {progress:.1f}% | Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")
        print(f"📈 Loss: {logs.get('loss', 0):.4f} | Acc: {logs.get('accuracy', 0):.4f} | Val_Loss: {logs.get('val_loss', 0):.4f} | Val_Acc: {logs.get('val_accuracy', 0):.4f}")

checkpoint = ModelCheckpoint("./model_6_classes.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

early_stopping = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_learningrate = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

epochs = 48
progress_callback = ProgressCallback(epochs)
callbacks_list = [early_stopping,checkpoint,reduce_learningrate,progress_callback]

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_set,
                   steps_per_epoch=max(1, train_set.n//train_set.batch_size),
                   epochs=epochs,
                   validation_data=test_set,
                   validation_steps=max(1, test_set.n//test_set.batch_size),
                   callbacks=callbacks_list)

# Plot training history
plt.style.use('dark_background')

plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()

print("Model trained on 6 emotion classes (excluding disgust):")
print("Classes:", emotion_classes)