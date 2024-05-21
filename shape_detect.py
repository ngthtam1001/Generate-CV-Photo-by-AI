import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import face_recognition

# Load the image and detect faces
image_path = "face1.jpg"

def calculate(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)

    # Load the predictor
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)

    # Convert to OpenCV format
    cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    original = cv_image.copy()

    # Process each face detected
    for (top, right, bottom, left) in face_locations:
        # Define the rectangle for dlib
        dlib_rect = dlib.rectangle(left, top, right, bottom)
        detected_landmarks = predictor(cv_image, dlib_rect).parts()

        landmarks = np.array([[p.x, p.y] for p in detected_landmarks])

        # Measurements
        # Forehead Length (approximate using between brows to top of the rectangle as hairline isn't marked)
        forehead_length = landmarks[27][1] - top  # Point 27 is between the brows

        # Face Width (between points 0 and 16, which are at the edge of the face horizontally)
        face_width = landmarks[16][0] - landmarks[0][0]

        # Jawline Length (from point 8 to midway between points 5 and 7, approximating jaw curve length)
        jawline_length = np.sqrt((landmarks[8][0] - (landmarks[5][0] + landmarks[7][0]) // 2) ** 2 + 
                                (landmarks[8][1] - (landmarks[5][1] + landmarks[7][1]) // 2) ** 2)

        # Face Length (from point 27 to point 8, which are the top of the forehead and chin)
        face_length = landmarks[8][1] - landmarks[27][1]
        
        return (forehead_length,face_width, jawline_length, face_length)