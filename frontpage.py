import sys
import cv2
import os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

width = 500
height = "auto"

class FrontPage(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):

        browse_button = QPushButton("Upload Photo", self)
        browse_button.setMinimumSize(100, 40)
        browse_button.setMaximumSize(100, 40)
        browse_button.clicked.connect(self.browse_image)
        
        self.predict_button = QPushButton("Predict Photo", self)
        self.predict_button.setMinimumSize(100, 40)
        self.predict_button.setMaximumSize(100, 40)
        self.predict_button.hide()

        self.video_button = QPushButton("Upload Video", self)
        self.video_button.setMinimumSize(100, 40)
        self.video_button.setMaximumSize(100, 40)
        self.video_button.clicked.connect(self.upload_video)
        
        camera_button = QPushButton("Turn on Camera", self)
        camera_button.setMinimumSize(100, 40)
        camera_button.setMaximumSize(100, 40)
        camera_button.clicked.connect(self.camera_on)

        past_result_button = QPushButton("Past Detections", self)
        past_result_button.setMinimumSize(100, 40)
        past_result_button.setMaximumSize(100, 40)
        past_result_button.clicked.connect(self.select_folder)

        # create a button to move to the previous image
        self.prev_button = QPushButton('Prev', self)
        self.prev_button.hide()
        self.prev_button.clicked.connect(self.show_prev_image)

        # create a button to move to the next image
        self.next_button = QPushButton('Next', self)
        self.next_button.hide()
        self.next_button.clicked.connect(self.show_next_image)

        vbox = QVBoxLayout()
        vbox.addWidget(browse_button)
        vbox.addWidget(self.video_button)
        vbox.addWidget(camera_button)
        vbox.addWidget(past_result_button)

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(400, 400)
        self.image_label.setScaledContents(True)
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.prev_button)
        vbox.addWidget(self.next_button)

        self.setLayout(vbox)
        self.setWindowTitle("Criminal Identification System (CIS)")

        # Set the size policy for the main widget to expanding in the vertical direction
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        
        # list to hold image file names
        self.image_files = []
        self.current_image_index = -1


    # Browse images from the computer to be uploaded on the app for prediction.
    def browse_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "Images (*.png *.xpm *.jpg *.bmp *.gif *jpeg)", options=options)
        if file_name:
            image_path = file_name
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap)
            self.predict_button.clicked.connect(self.predict_photo(image_path))
        

    # Browse videos from the computer to be uploaded on the app for prediction.
    def upload_video(self):
        video_options = QFileDialog.Options()
        video_options = QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "Videos (*.mp4)", options=video_options)
        if file_name:
            print(file_name)
            video_path = file_name
            self.video_button.clicked.connect(self.predict_video(video_path))


    # Turn on the App camera to detect faces.
    def camera_on(self):    
        from camera_predictor import detectImages
        detectImages.turn_on_camera(detectImages)
        return None


    # Predict the face on the image uploaded on the app.
    def predict_photo(self, photo):    
        from picture_detector import Picture_Predictor
        Picture_Predictor.predict_image(self, photo)
        return None
    
    # Predict the faces on the frames of the video uploaded
    def predict_video(self, video):
        from video_extractor import Predict_video
        # from vid2 import Predict_video
        Predict_video.main(self, video)
        return None


    def select_folder(self):
        # open a file dialog to select a folder
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folder_name = QFileDialog.getExistingDirectory(self, "Select Folder", options=options)
        if folder_name:
            # get a list of all image files in the folder
            self.image_files = [os.path.join(folder_name, file) for file in os.listdir(folder_name)
                                if file.endswith(('png', 'jpg', 'bmp'))]
            if self.image_files:
                # enable the next and prev buttons
                self.next_button.show()
                self.prev_button.show()
                # show the first image
                self.current_image_index = 0
                self.show_image(self.current_image_index)


    def show_image(self, index):
        # read the image file using OpenCV
        file_name = self.image_files[index]
        image = cv2.imread(file_name)

        # convert the image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # display the image on the label
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qImg))
        
        # disable the prev button if we're on the first image
        if self.current_image_index == 0:
                self.prev_button.hide()
        elif self.current_image_index > 0:
            self.prev_button.show()


    def show_prev_image(self):
        # show the previous image in the list
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image(self.current_image_index)

            # disable the prev button if we're on the first image
            if self.current_image_index == 0:
                self.prev_button.hide()

        # enable the next button if it's disabled
        if not self.next_button.show():
            self.next_button.show()


    def show_next_image(self):
        # show the previous image in the list
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.show_image(self.current_image_index)
        elif self.current_image_index == len(self.image_files) - 1:
            self.next_button.hide()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    frontPage = FrontPage()
    frontPage.show()
    sys.exit(app.exec_())
