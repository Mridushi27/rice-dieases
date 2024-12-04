# Import necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the seed for reproducibility
np.random.seed(42)

# Define the CNN model architecture
def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load the dataset
train_dir = 'dataset/train/'
test_dir = 'dataset/test/'

# Define the data generators for training and testing
train_datagen = ImageDataGenerator(rescale=1./255,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                            target_size=(256, 256),
                                                            batch_size=32,
                                                            class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                          target_size=(256, 256),
                                                          batch_size=32,
                                                          class_mode='categorical')

# Create the CNN model
model = create_cnn_model()

# Train the model
history = model.fit(train_generator,
                      epochs=10,
                      validation_data=test_generator)

# Plot the training and validation accuracy and loss
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc:.2f}')
print(f'Test Loss: {test_loss:.2f}')

# Use the model to make predictions on new data
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Print the classification report and confusion matrix
print('Classification Report:')
print(classification_report(test_generator.classes, predicted_classes))
print('Confusion Matrix:')
print(confusion_matrix(test_generator.classes, predicted_classes))
