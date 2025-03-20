import numpy as np
import os
from deepface import DeepFace
import cv2 as cv

# Paths and variables
datasetPath = r"C:\Users\u\Desktop\archive2"
data = []
emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

total_images = 0
faces_detected = 0
faces_not_detected = 0

for emotion in os.listdir(datasetPath):
    emotionPath = os.path.join(datasetPath, emotion)

    for image_name in os.listdir(emotionPath):
        imagePath = os.path.join(emotionPath, image_name)
        total_images += 1

        try:
            img = cv.imread(imagePath)
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            # Use DeepFace to detect faces
            faces = DeepFace.extract_faces(img_rgb, detector_backend="opencv", enforce_detection=False) #mtcnn de kullanilabilir

            if faces:
                for face in faces:
                    x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']
                    
                    if w > 0 and h > 0:
                        faces_detected += 1
                        
                        # Crop the detected face
                        face_img = img[y:y+h, x:x+w]

                        # Resize to 48x48 for CNN input
                        face_resized = cv.resize(face_img, (48, 48))

                        # Convert to grayscale
                        face_gray = cv.cvtColor(face_resized, cv.COLOR_RGB2GRAY)

                        # Store processed face and label
                        data.append((face_gray, emotions.index(emotion)))
            else:
                faces_not_detected += 1

        except Exception as e:
            print(f"Error processing {imagePath}: {str(e)}")
            faces_not_detected += 1

# Summary
print(f"\nProcessing Complete!")
print(f"Total Images: {total_images}")
print(f"Faces Detected: {faces_detected}")
print(f"Faces Not Detected: {faces_not_detected}")
print(f"Detection Rate: {(faces_detected/total_images)*100:.2f}%")

# Convert data to NumPy arrays
X = np.array([item[0] for item in data]).reshape(-1, 48, 48, 1) / 255.0  # Normalize pixel values
y = np.array([item[1] for item in data])  # Labels (emotion classes)

# Save dataset
np.save("X.npy", X)
np.save("y.npy", y)

print("\nDataset saved successfully!")
