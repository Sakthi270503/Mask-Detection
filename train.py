import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

def load_dataset(dataset_path):
    images = []
    labels = []
    
    class_names = sorted(os.listdir(dataset_path))  # Get list of class names
    
    for class_index, class_name in enumerate(class_names):
        label_path = os.path.join(dataset_path, class_name)
        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (128, 128))  # Resize image to a common size
            image = image.astype('float32') / 255.0  # Normalize pixel values
            images.append(image)
            labels.append(class_index)  # Assign label based on class index
    
    return np.array(images), np.array(labels)

def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Output probabilities for each class
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(images, labels):
    input_shape = images[0].shape
    num_classes = len(np.unique(labels))  # Get number of unique classes
    
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    model = create_model(input_shape, num_classes)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    return model

def evaluate_model(model, images, labels):
    loss, accuracy = model.evaluate(images, labels)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    # Step 0: Define dataset path
    dataset_path = r"C:\Users\Sakthi Murugan V\OneDrive\Desktop\m-datasets\data"
    
    # Step 1: Load dataset
    images, labels = load_dataset(dataset_path)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Create and train model
    model = create_model(X_train[0].shape, len(np.unique(labels)))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Save trained model to HDF5 file
    model.save("trained_model.h5")

