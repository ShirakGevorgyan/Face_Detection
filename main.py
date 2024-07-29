import cv2
import os
import pickle
import numpy as np

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

label_list = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

with open('mymodel.pkl', 'rb') as file:
    model = pickle.load(file)

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Create a directory to save cropped faces
if not os.path.exists('cropped_faces'):
    os.makedirs('cropped_faces')

face_id = 0  # Initialize a face ID to save images with unique names

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces and save cropped faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Crop the face from the frame
        cropped_face = gray[y:y+h, x:x+w]
        # Resize the cropped face to the required input size of the model (48x48)
        cropped_face = cv2.resize(cropped_face, (48, 48))
        # Convert to 3 channels by duplicating the grayscale channel
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_GRAY2BGR)
        print(cropped_face.shape)  # This should print (48, 48, 3)
        # Normalize the pixel values
        cropped_face = cropped_face.astype('float32') / 255
        # Expand dimensions to match the model input shape
        cropped_face = np.expand_dims(cropped_face, axis=0)
        # Predict the emotion
        pred = model.predict(cropped_face)
        emotion_label = label_list[np.argmax(pred)]
        # Display the emotion label within the rectangle
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
