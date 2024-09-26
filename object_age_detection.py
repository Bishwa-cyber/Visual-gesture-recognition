# import cv2
# import numpy as np
# from deepface import DeepFace

# # Load pre-trained object detection model (MobileNet SSD)
# net = cv2.dnn.readNetFromCaffe(
#     'deploy.prototxt', 
#     'res10_300x300_ssd_iter_140000.caffemodel'
# )

# # Load pre-trained age estimation model
# # (Here we use DeepFace library for simplicity)
# def estimate_age(image):
#     try:
#         analysis = DeepFace.analyze(image, actions=['age'])
#         return analysis[0]['age']
#     except Exception as e:
#         print(f"Error in age estimation: {e}")
#         return "Unknown"

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     (h, w) = frame.shape[:2]

#     # Prepare the image for object detection
#     blob = cv2.dnn.blobFromImage(
#         cv2.resize(frame, (300, 300)), 
#         1.0, (300, 300), (104.0, 177.0, 123.0)
#     )
#     net.setInput(blob)
#     detections = net.forward()

#     for i in range(0, detections.shape[2]):
#         confidence = detections[0, 0, i, 2]

#         if confidence > 0.5:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
            
#             # Draw bounding box
#             cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            
#             # Crop the detected face for age estimation
#             face = frame[startY:endY, startX:endX]
#             age = estimate_age(face)
            
#             # Display the age
#             cv2.putText(
#                 frame, f"Age: {age}", 
#                 (startX, startY - 10), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 
#                 0.9, (0, 255, 0), 2
#             )

#     # Show the frame
#     cv2.imshow('Object and Age Detection', frame)

#     # Exit on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()










# import cv2
# import numpy as np
# from deepface import DeepFace

# # Load pre-trained object detection model (MobileNet SSD)
# net = cv2.dnn.readNetFromCaffe(
#     'C:/Users/ASUS/OneDrive/Desktop/opencv/deploy.prototxt', 
#     'C:/Users/ASUS/OneDrive/Desktop/opencv/res10_300x300_ssd_iter_140000.caffemodel'
# )

# # Load pre-trained age estimation model
# def estimate_age(image):
#     try:
#         analysis = DeepFace.analyze(image, actions=['age'], enforce_detection=False)
#         return analysis['age']
#     except Exception as e:
#         print(f"Error in age estimation: {e}")
#         return "Unknown"

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     (h, w) = frame.shape[:2]

#     # Prepare the image for object detection
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
#     net.setInput(blob)
#     detections = net.forward()

#     for i in range(0, detections.shape[2]):
#         confidence = detections[0, 0, i, 2]

#         if confidence > 0.5:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")

#             # Draw bounding box
#             cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

#             # Crop the detected face for age estimation
#             face = frame[startY:endY, startX:endX]
#             age = estimate_age(face)

#             # Display the age
#             cv2.putText(frame, f"Age: {age}", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     # Show the frame
#     cv2.imshow('Object and Age Detection', frame)

#     # Exit on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# from deepface import DeepFace

# # Load pre-trained object detection model (MobileNet SSD)
# net = cv2.dnn.readNetFromCaffe(
#     'deploy.prototxt', 
#     'res10_300x300_ssd_iter_140000.caffemodel'
# )

# # Function to estimate age and gender
# def estimate_age_gender(image):
#     try:
#         # Using DeepFace for both age and gender analysis
#         analysis = DeepFace.analyze(image, actions=['age', 'gender'])
#         age = analysis['age']
#         gender = analysis['gender']
#         return age, gender
#     except Exception as e:
#         print(f"Error in age/gender estimation: {e}")
#         return "Unknown", "Unknown"

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     (h, w) = frame.shape[:2]

#     # Prepare the image for object detection
#     blob = cv2.dnn.blobFromImage(
#         cv2.resize(frame, (300, 300)), 
#         1.0, (300, 300), (104.0, 177.0, 123.0)
#     )
#     net.setInput(blob)
#     detections = net.forward()

#     for i in range(0, detections.shape[2]):
#         confidence = detections[0, 0, i, 2]

#         if confidence > 0.5:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
            
#             # Draw bounding box
#             cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            
#             # Crop the detected face for age and gender estimation
#             face = frame[startY:endY, startX:endX]
#             age, gender = estimate_age_gender(face)
            
#             # Display the age and gender
#             text = f"Age: {age}, Gender: {gender}"
#             cv2.putText(
#                 frame, text, 
#                 (startX, startY - 10), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 
#                 0.9, (0, 255, 0), 2
#             )

#     # Show the frame
#     cv2.imshow('Age and Gender Detection', frame)

#     # Exit on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()





# import cv2
# import numpy as np
# from deepface import DeepFace

# # Load pre-trained object detection model (MobileNet SSD)
# net = cv2.dnn.readNetFromCaffe(
#     'C:/Users/ASUS/OneDrive/Desktop/opencv/deploy.prototxt', 
#     'C:/Users/ASUS/OneDrive/Desktop/opencv/res10_300x300_ssd_iter_140000.caffemodel'
# )

# # Load pre-trained age estimation model
# def estimate_age_and_gender(image):
#     try:
#         analysis = DeepFace.analyze(image, actions=['age', 'gender'], enforce_detection=False)
#         age = analysis['age']
#         gender = analysis['gender']
#         return f"Age: {age}", f"Gender: {gender}"
#     except Exception as e:
#         print(f"Error in age/gender estimation: {e}")
#         return "Age: Unknown", "Gender: Unknown"

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     (h, w) = frame.shape[:2]

#     # Prepare the image for object detection
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
#     net.setInput(blob)
#     detections = net.forward()

#     for i in range(0, detections.shape[2]):
#         confidence = detections[0, 0, i, 2]

#         if confidence > 0.5:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")

#             # Draw bounding box
#             cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

#             # Crop the detected face for age and gender estimation
#             face = frame[startY:endY, startX:endX]
#             age, gender = estimate_age_and_gender(face)

#             # Display the age and gender
#             cv2.putText(frame, age, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
#             cv2.putText(frame, gender, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

#     # Show the frame
#     cv2.imshow('Object, Age, and Gender Detection', frame)

#     # Exit on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# from deepface import DeepFace
# import mediapipe as mp

# # Load pre-trained object detection model (MobileNet SSD)
# net = cv2.dnn.readNetFromCaffe(
#     'C:/Users/ASUS/OneDrive/Desktop/opencv/deploy.prototxt', 
#     'C:/Users/ASUS/OneDrive/Desktop/opencv/res10_300x300_ssd_iter_140000.caffemodel'
# )

# # Initialize mediapipe hands model
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
# mp_drawing = mp.solutions.drawing_utils

# # Load pre-trained age and gender estimation model
# def estimate_age_and_gender(image):
#     try:
#         # Ensure image is in RGB format
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         analysis = DeepFace.analyze(image_rgb, actions=['age', 'gender'], enforce_detection=False)
#         # Extract the first result (if multiple faces are detected)
#         if len(analysis) > 0:
#             age = analysis[0]['age']
#             gender = analysis[0]['gender']
#             return f"Age: {age}", f"Gender: {gender}"
#         else:
#             return "Age: Unknown", "Gender: Unknown"
#     except Exception as e:
#         print(f"Error in age/gender estimation: {e}")
#         return "Age: Unknown", "Gender: Unknown"

# # Gesture detection function
# def detect_gestures(image):
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = hands.process(image_rgb)
#     action = "No Gesture"

#     if results.multi_hand_landmarks:
#         for landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)
#             # Simple gesture detection logic, e.g., detecting hand waving or shaking
#             # You can enhance this part with specific logic for different gestures
#             action = "Gesture Detected"  # Placeholder for actual gesture recognition

#     return action

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     (h, w) = frame.shape[:2]

#     # Prepare the image for object detection
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
#     net.setInput(blob)
#     detections = net.forward()

#     for i in range(0, detections.shape[2]):
#         confidence = detections[0, 0, i, 2]

#         if confidence > 0.5:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")

#             # Draw bounding box
#             cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

#             # Crop the detected face for age and gender estimation
#             face = frame[startY:endY, startX:endX]
            
#             # Resize the face image to (224, 224) if needed
#             face_resized = cv2.resize(face, (224, 224))
            
#             age, gender = estimate_age_and_gender(face_resized)

#             # Display the age and gender
#             cv2.putText(frame, age, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
#             cv2.putText(frame, gender, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

#     # Detect gestures
#     action = detect_gestures(frame)

#     # Display the detected action
#     cv2.putText(frame, f"Action: {action}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

#     # Show the frame
#     cv2.imshow('Object, Age, Gender, and Gesture Detection', frame)

#     # Exit on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()







# 1. Import Libraries
# python
# Copy code
# import cv2
# import numpy as np
# from deepface import DeepFace
# import mediapipe as mp
# cv2: OpenCV library used for computer vision tasks like image processing.
# numpy: Library for numerical operations, particularly useful for handling arrays.
# DeepFace: A library for deep learning-based face analysis, including age and gender detection.
# mediapipe: A library for real-time hand tracking and gesture recognition.
# 2. Load Pre-trained Object Detection Model
# python
# Copy code
# net = cv2.dnn.readNetFromCaffe(
#     'C:/Users/ASUS/OneDrive/Desktop/opencv/deploy.prototxt', 
#     'C:/Users/ASUS/OneDrive/Desktop/opencv/res10_300x300_ssd_iter_140000.caffemodel'
# )
# cv2.dnn.readNetFromCaffe: Loads a pre-trained object detection model based on the Caffe framework.
# deploy.prototxt: Configuration file for the MobileNet SSD model.
# res10_300x300_ssd_iter_140000.caffemodel: The pre-trained weights file for the model.
# 3. Initialize Mediapipe Hands Model
# python
# Copy code
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
# mp_drawing = mp.solutions.drawing_utils
# mp_hands.Hands(): Initializes the hand detection model from Mediapipe.
# mp_drawing: Utility to draw landmarks and connections on the image.
# 4. Define Age and Gender Estimation Function
# python
# Copy code
# def estimate_age_and_gender(image):
#     try:
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         analysis = DeepFace.analyze(image_rgb, actions=['age', 'gender'], enforce_detection=False)
#         if len(analysis) > 0:
#             age = analysis[0]['age']
#             gender = analysis[0]['gender']
#             return f"Age: {age}", f"Gender: {gender}"
#         else:
#             return "Age: Unknown", "Gender: Unknown"
#     except Exception as e:
#         print(f"Error in age/gender estimation: {e}")
#         return "Age: Unknown", "Gender: Unknown"
# cv2.cvtColor: Converts the image from BGR to RGB format as required by the DeepFace library.
# DeepFace.analyze: Analyzes the image for age and gender. The actions parameter specifies what to analyze.
# enforce_detection=False: Allows processing even if the face detection is not strict.
# Exception Handling: Catches and prints any errors during analysis.
# 5. Define Gesture Detection Function
# python
# Copy code
# def detect_gestures(image):
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = hands.process(image_rgb)
#     action = "No Gesture"

#     if results.multi_hand_landmarks:
#         for landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)
#             action = "Gesture Detected"  # Placeholder for actual gesture recognition

#     return action
# hands.process: Processes the image to detect hand landmarks.
# mp_drawing.draw_landmarks: Draws the hand landmarks and connections on the image.
# action: Placeholder for recognizing specific gestures. This could be enhanced with more specific logic.
# 6. Initialize Webcam and Start Processing
# python
# Copy code
# cap = cv2.VideoCapture(0)
# cv2.VideoCapture(0): Initializes the webcam. 0 refers to the default camera.
# 7. Main Loop
# python
# Copy code
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     (h, w) = frame.shape[:2]

#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
#     net.setInput(blob)
#     detections = net.forward()

#     for i in range(0, detections.shape[2]):
#         confidence = detections[0, 0, i, 2]

#         if confidence > 0.5:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")

#             cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

#             face = frame[startY:endY, startX:endX]
#             face_resized = cv2.resize(face, (224, 224))
            
#             age, gender = estimate_age_and_gender(face_resized)

#             cv2.putText(frame, age, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
#             cv2.putText(frame, gender, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

#     action = detect_gestures(frame)
#     cv2.putText(frame, f"Action: {action}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

#     cv2.imshow('Object, Age, Gender, and Gesture Detection', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# Frame Capture: Reads each frame from the webcam.
# Object Detection: Prepares the frame for object detection, detects objects, and draws bounding boxes around detected faces.
# Face Cropping and Estimation: Crops the detected face, resizes it, and estimates age and gender.
# Gesture Detection: Processes the frame for hand gestures and updates the action.
# Display: Shows the processed frame with detected objects, age, gender, and gestures.
# Exit: Breaks the loop and closes the application if 'q' is pressed.
# 8. Release Resources
# python
# Copy code
# cap.release()
# cv2.destroyAllWindows()
# cap.release(): Releases the webcam.
# cv2.destroyAllWindows(): Closes all OpenCV windows.
# Summary
# Object Detection: Detects faces in the video feed.
# Age and Gender Estimation: Analyzes detected faces to estimate age and gender.
# Gesture Detection: Recognizes hand gestures and displays detected actions.
# Display: Shows results and updates in real-time.
# This breakdown should help you explain each part of
















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















