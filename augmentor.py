import keras
import tensorflow as tf
import numpy as np
from skimage import io
import os

from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

# Construct an instance of the ImageDataGenerator class
# Pass the augmentation parameters through the constructor. 

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=45, #random rotation between 0 to 45
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=None,
    shear_range=0.2,
    zoom_range=0.2,
    channel_shift_range=0.0,
    fill_mode='reflect',
    cval=125,
    horizontal_flip=True,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0,
    interpolation_order=1,
    dtype=None)

lstcat=['chris_evans','chris_hemsworth','mark_ruffalo','robert_downey_jr','scarlett_johansson']
aug = ['chris_evans_aug','chris_hemsworth_aug','mark_ruffalo_aug','robert_downey_aug','scarlett_johansson_aug']
aug_dir = 'Avengers Dataset/images/new_test/'

k=0


for item in lstcat:
   
    image_directory = 'Datasets/Avengers Dataset/images/test/'+item+'/'
    
    SIZE = 128
    dataset = []

    my_images = os.listdir(image_directory)
    for i, image_name in enumerate(my_images):
        if (image_name.split('.')[1] == 'jpg' or image_name.split('.')[1] == 'png'):
            image = io.imread(image_directory + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((SIZE,SIZE))
            dataset.append(np.array(image))

    x = np.array(dataset)

    #Let us save images to get a feel for the augmented images.
    #Create an iterator either by using image dataset in memory (using flow() function)
    #or by using image dataset from a directory (using flow_from_directory)
    #from directory can beuseful if subdirectories are organized by class
    
    i = 1
    for batch in datagen.flow_from_directory(directory='Datasets/Avengers Dataset/images/test/', 
                                         batch_size=32,  
                                         target_size=(256, 256),
                                         color_mode="rgb",
                                         save_to_dir='Datasets/Avengers Dataset/images/new_test/'+aug[k]+'/', 
                                         save_prefix='aug', 
                                         save_format='png'):
        i += 1
        if i > 300:
            k=k+1
            break   # otherwise the generator would loop indefinitely  