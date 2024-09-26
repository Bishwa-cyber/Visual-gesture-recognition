import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
from datetime import datetime
import pytz

# Load pre-trained object detection model (MobileNet SSD)
net = cv2.dnn.readNetFromCaffe(
    'C:/Users/ASUS/OneDrive/Desktop/opencv/deploy.prototxt', 
    'C:/Users/ASUS/OneDrive/Desktop/opencv/res10_300x300_ssd_iter_140000.caffemodel'
)

# Initialize mediapipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Load pre-trained age and gender estimation model
def estimate_age_and_gender(image):
    try:
        # Ensure image is in RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        analysis = DeepFace.analyze(image_rgb, actions=['age', 'gender'], enforce_detection=False)
        # Extract the first result (if multiple faces are detected)
        if len(analysis) > 0:
            age = analysis[0]['age']
            gender = analysis[0]['gender']
            return f"Age: {age}", f"Gender: {gender}"
        else:
            return "Age: Unknown", "Gender: Unknown"
    except Exception as e:
        print(f"Error in age/gender estimation: {e}")
        return "Age: Unknown", "Gender: Unknown"

# Gesture detection function
def detect_gestures(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    action = "No Gesture"

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)
            # Detect simple gestures based on hand landmarks positions
            for hand_landmark in results.multi_hand_landmarks:
                if hand_landmark.landmark[mp_hands.HandLandmark.WRIST].y > hand_landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y:
                    action = "Hand Waving"
                else:
                    action = "Hand Closed"

    return action

# Function to get real-time clock of Kolkata timezone
def get_kolkata_time():
    kolkata_tz = pytz.timezone('Asia/Kolkata')
    kolkata_time = datetime.now(kolkata_tz)
    return kolkata_time.strftime('%H:%M:%S')

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Prepare the image for object detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Crop the detected face for age and gender estimation
            face = frame[startY:endY, startX:endX]
            
            # Resize the face image to (224, 224) if needed
            face_resized = cv2.resize(face, (224, 224))
            
            age, gender = estimate_age_and_gender(face_resized)

            # Display the age and gender
            cv2.putText(frame, age, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(frame, gender, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # Get the current time in Kolkata timezone and display it under the green box
            kolkata_time = get_kolkata_time()
            cv2.putText(frame, f"Time: {kolkata_time}", (startX, endY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # Detect gestures
    action = detect_gestures(frame)

    # Display the detected action
    cv2.putText(frame, f"Action: {action}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # Show the frame
    cv2.imshow('Object, Age, Gender, Gesture, and Time Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()















