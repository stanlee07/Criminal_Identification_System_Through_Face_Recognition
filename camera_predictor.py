import cv2
import numpy as np
import os
from datetime import timedelta
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog
from PyQt5.QtCore import Qt
import keras.utils as image
from keras.models import load_model
from datetime import datetime, date


image_width = 64
image_height = 64


# Load the trained model
model = load_model('identification-model.h5')

now = datetime.now()
dt_string = now.strftime("%H.%M.%S.%f")

# Classes
class_labels = ['Chris Evans', 'Chris Hemsworth', 'Mark Ruffalo', 'Robert Downey Jr', 'Scarlett Johansson']

# Set the face detection classifier
face_cascade = cv2.CascadeClassifier("Packages/cv2/data/haarcascade_frontalface_default.xml")

class detectImages():

    def __init__(self) -> None:
        super().__init__()

        self.turn_on_camera()

    
    # Turns on the camera/webcam and detects the faces.
    def turn_on_camera(self):
        
        folder = "Datasets/Detected Faces/Live Feeds"
        os.chdir(folder)
        
        print (os.getcwd())

        today = date.today()
        today = today.strftime("%b-%d-%Y")
        filename = today +' Images'

        # make a folder by the name of the video file
        if not os.path.isdir(filename):
            os.mkdir(filename)

        # read the video file
        cap = cv2.VideoCapture(0)

        # start the loop
        count = 0
        while True:

            # Capture the video frame by frame
            ret, frame = cap.read()

            # Detect faces
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=3)


            # Draw a rectangle around the first face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Crop the face from the frame
                face = frame[y:y+h, x:x+w]

                # verify the detected ROI is actually a face
                if w > 0 and h > 0:
                    
                    # increment the frame count
                    count += 1

                    # Resize the face to the desired dimensions
                    face_resized = cv2.resize(face, (image_width, image_height))

                    # Predict the detected faces
                    face_resized = image.img_to_array(face_resized)
                    face_resized = np.expand_dims(face_resized, axis=0)

                    # Make a prediction using the loaded model
                    result = model.predict(face_resized)
                    prediction = class_labels[result.argmax()]


                    # Drawing a bounding box around the detected face and display the predicted label
                    cv2.rectangle(face, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Name: {prediction}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

                    # Save the face to a file
                    cv2.imwrite(os.path.join(filename, f"{dt_string}-{count}.png"), frame)

            # Display the resulting frame
            cv2.imshow('Criminal Identification System (CIS)', frame)

            # The 'q' button is set as the quitting button.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        
        # After the loop release the cap object
        cap.release()

        # Destroy all the windows
        cv2.destroyAllWindows()



