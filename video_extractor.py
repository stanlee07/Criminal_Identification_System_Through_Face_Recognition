from datetime import timedelta
import cv2
import numpy as np
import os
import keras.utils as image
import time
from keras.models import load_model


width = 64
height = 64


# Initialize a counter for the frames processed
frame_num = 0

# Set the face detection classifier
face_cascade = cv2.CascadeClassifier("Packages/cv2/data/haarcascade_frontalface_default.xml")


# Load the pre-trained face recognition model
model = load_model('identification-model.h5')

# Classes
class_labels = ['Chris Evans', 'Chris Hemsworth', 'Mark Ruffalo', 'Robert Downey Jr', 'Scarlett Johansson']



class Predict_video():

    def main(self,video_file):
        
        # Location to save the pictures.
        folder = "Datasets/Detected Faces/Uploaded Videos"
        os.chdir(folder)

        filename = os.path.splitext(os.path.basename(video_file))[0]
        filename += ' Images'

        # make a folder by the name of the video file
        if not os.path.isdir(filename):
            os.mkdir(filename)

        # read the video file    
        cap = cv2.VideoCapture(video_file)

        # Initialize a counter for the frames processed
        frame_num = 1

        # Set the time to start capturing frames
        start_time = time.time()


        while True:
            
            #  Read the next frame from the video
            is_read, frame = cap.read()
            if not is_read:
                # break out of the loop if there are no frames to read
                break

            # Detect faces
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
    
            # Get the current time
            current_time = time.time()

            # If one second has passed since the last frame capture, save the current frame
            if current_time - start_time >= 0.5:    
                
                # Draw a rectangle around the first face
                for (x, y, w, h) in faces[:]:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Crop the face from the frame
                    face = frame[y:y+h, x:x+w]

                    # verify the detected ROI is actually a face
                    if w > 0 and h > 0:

                        # Resize the face to the desired dimensions
                        face = cv2.resize(face, (width, height))

                        # Preprocess the face image
                        facer = cv2.resize(face, (64, 64))
                        facer = image.img_to_array(facer)
                        facer = np.expand_dims(facer, axis=0)

                        # Obtain the face embedding
                        result = model.predict(facer)

                        # Compare the face embedding with known embeddings in your database and output the matching result
                        # Get the predicted class label
                        prediction = class_labels[result.argmax()]


                        # Drawing a bounding box around the detected face and display the predicted label
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"Name: {prediction}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

                    # Save the face to a file
                    cv2.imwrite(os.path.join(filename, f"frame{frame_num}.jpg"), frame)
                    frame_num += 1
                
                # Reset the start time
                start_time = current_time
        
        
        # Release the input video file and exit
        cap.release()
        cv2.destroyAllWindows()
