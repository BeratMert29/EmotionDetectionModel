import numpy as np
import os
import cv2 as cv
from deepface import DeepFace
import logging

# Configuration
TRAIN_PATH = r"C:\Users\u\Downloads\archive(3)\train"
TEST_PATH = r"C:\Users\u\Downloads\archive(3)\test"
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = 48

# Set up logging for errors
logging.basicConfig(filename="error_log.txt", level=logging.ERROR, format='%(asctime)s - %(message)s')

def process_data(data_path, data_type):
    data = []
    total_images = 0
    faces_detected = 0
    faces_not_detected = 0

    print(f"\nProcessing {data_type} images...")

    for emotion in os.listdir(data_path):
        if emotion.startswith('.') or not os.path.isdir(os.path.join(data_path, emotion)):
            continue

        emotion_path = os.path.join(data_path, emotion)
        print(f"Processing {emotion} images...")

        for image_name in os.listdir(emotion_path):
            if image_name.startswith('.'):
                continue

            image_path = os.path.join(emotion_path, image_name)
            total_images += 1

            try:
                img = cv.imread(image_path)
                if img is None:
                    print(f"Could not read image: {image_path}")
                    faces_not_detected += 1
                    continue

                img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

                # Use DeepFace for face detection
                faces = DeepFace.extract_faces(img_rgb, detector_backend="opencv", enforce_detection=False)
                
                if faces:
                    face = faces[0]
                    x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']
                    
                    if w > 0 and h > 0:
                        face_img = img[y:y+h, x:x+w]
                        face_resized = cv.resize(face_img, (IMG_SIZE, IMG_SIZE))
                        face_gray = cv.cvtColor(face_resized, cv.COLOR_RGB2GRAY)
                        face_normalized = face_gray / 255.0
                        face_processed = face_normalized  # Shape will be (48, 48), no extra channel

                        faces_detected += 1
                        data.append((face_processed, emotions.index(emotion)))
                    else:
                        faces_not_detected += 1
                else:
                    faces_not_detected += 1

            except Exception as e:
                logging.error(f"Error processing {image_path}: {str(e)}")
                faces_not_detected += 1

    return data, total_images, faces_detected, faces_not_detected

# Check if directories exist
if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"Training directory not found at {TRAIN_PATH}")
if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"Test directory not found at {TEST_PATH}")

# Process training data
train_data, train_total, train_detected, train_not_detected = process_data(TRAIN_PATH, "training")
if train_data:
    X_train = np.array([item[0] for item in train_data])
    y_train = np.array([item[1] for item in train_data])
    np.save("train_X.npy", X_train)
    np.save("train_y.npy", y_train)
    print("\nTraining dataset saved successfully!")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

# Process test data
test_data, test_total, test_detected, test_not_detected = process_data(TEST_PATH, "test")
if test_data:
    X_test = np.array([item[0] for item in test_data])
    y_test = np.array([item[1] for item in test_data])
    np.save("test_X.npy", X_test)
    np.save("test_y.npy", y_test)
    print("\nTest dataset saved successfully!")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

# Print summary
print("\nProcessing Summary:")
print("\nTraining Data:")
print(f"Total Images: {train_total}")
print(f"Faces Detected: {train_detected}")
print(f"Faces Not Detected: {train_not_detected}")
print(f"Detection Rate: {(train_detected/train_total)*100:.2f}%")

print("\nTest Data:")
print(f"Total Images: {test_total}")
print(f"Faces Detected: {test_detected}")
print(f"Faces Not Detected: {test_not_detected}")
print(f"Detection Rate: {(test_detected/test_total)*100:.2f}%")
