import cv2
import numpy as np
import keras.utils as image
from keras.models import load_model
import os
import pandas as pd


# Load the trained model
model = load_model('identification-model.h5')

# Classes
class_labels = ['Chris Evans', 'Chris Hemsworth', 'Mark Ruffalo', 'Robert Downey Jr', 'Scarlett Johansson']

# Set the face detection classifier
face_cascade = cv2.CascadeClassifier("Packages/cv2/data/haarcascade_frontalface_default.xml")

# image size
image_width = 64
image_height = 64

# Create an empty dataframe to store the metrics
metrics_df = pd.DataFrame(columns=['Photo', 'Loss', 'Accuracy'])


# Image detection class to predict the faces on the images sent to the Predict_Image function
class Picture_Predictor():
    def __init__(self):
        super().__init__()

        self.predict_image()

    def predict_image(self, photo):
        
        photo_name = os.path.splitext(os.path.basename(photo))[0]

        # Location to save the pictures.
        folder = "Datasets/Detected Faces/Pictures"

        # Reading the uploaded photo using CV2 for the face detection and preprocessing
        test_image = cv2.imread(photo)

        # Detect faces in the input image using the Haar cascades algorithm
        test_image_faces = face_cascade.detectMultiScale(test_image, scaleFactor=1.3, minNeighbors=5)

        # Loop over all detected faces and make a prediction on each one
        for (x, y, w, h) in test_image_faces:
            # Extract the face region from the input image and preprocess it
            face = test_image[y:y+h, x:x+w]

            # verify the detected ROI is actually a face
            if w > 0 and h > 0:

                face = cv2.resize(face, (image_width, image_height))
                face = image.img_to_array(face)
                face = np.expand_dims(face, axis=0)

                # Make a prediction using the loaded model
                result = model.predict(face)
                prediction = class_labels[result.argmax()]
                
                # Define a threshold to filter out uncertain predictions
                threshold = 0.9
                
                # Check if the predicted label is above the threshold
                if result.max() >= threshold:
                    # Drawing a bounding box around the detected face and display the predicted label
                    cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(test_image, f"Name: {prediction}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                    print (prediction)
                else:
                    # Drawing a bounding box around the detected face and display a message
                    cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(test_image, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                    print (prediction)

        # Save the face to a file
        cv2.imwrite(os.path.join(folder, f"{photo_name}.jpg"), test_image)

        # Display the output image
        cv2.imshow('Output', test_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

        