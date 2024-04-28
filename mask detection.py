import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(r'C:\Users\Sakthi Murugan V\OneDrive\Desktop\Mask-Detection\trained_model.h5')  # Replace 'path_to_your_trained_model.h5' with the actual path

# Define labels for mask and without mask
labels = {0: 'Without Mask', 1: 'With Mask'}

# Function to detect masks in a frame
def detect_masks(frame):
    # Preprocess the frame (resize, normalize pixel values)
    resized_frame = cv2.resize(frame, (128, 128))
    normalized_frame = resized_frame.astype('float32') / 255.0
    expanded_frame = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension
    
    # Predict mask/no-mask for the frame
    prediction = model.predict(expanded_frame)
    label_index = int(np.round(prediction)[0][0])  # Round prediction to nearest integer
    
    # Get the corresponding label
    label = labels[label_index]
    
    # Display the label on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

# Open a video capture object (0 for webcam, or path to video file)
cap = cv2.VideoCapture(0)  # Change 0 to the path of your video file if you want to detect masks in a video

# Loop to capture frames and detect masks
while True:
    ret, frame = cap.read()  # Read frame from video stream
    if not ret:
        break
    
    # Detect masks in the frame
    frame = detect_masks(frame)
    
    # Display the frame
    cv2.imshow('Mask Detection', frame)
    
    # Check for 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
