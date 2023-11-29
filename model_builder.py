# This file would augment the images in the trianing data and also train the model with the training data.

import numpy as np
import random
import os

import matplotlib
from matplotlib import pyplot as plt

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


# Stopping the backend from being interactive that is from panning and zooming.
# matplotlib.use('Agg')

# Image parameters
image_width = 64
image_height = 64
batch_size = 32


# Dataset folders
train_data = 'Datasets/Avengers Dataset/images/new_train'
test_data = 'Datasets/Avengers Dataset/images/new_test'
val_data = 'Datasets/Avengers Dataset/images/new_val'


# Importing the training dataset
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode='nearest')
training_set = train_datagen.flow_from_directory(train_data, target_size = (image_width, image_height), batch_size = batch_size,  class_mode = 'categorical')


# Importing the test dataset
test_datagen = ImageDataGenerator()
test_set = test_datagen.flow_from_directory(test_data, target_size = (image_width, image_height), batch_size = batch_size,  class_mode = 'categorical')


# Importing the validation dataset
val_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode='nearest')
val_set = val_datagen.flow_from_directory(val_data, target_size = (image_width, image_height), batch_size = batch_size,  class_mode = 'categorical')


# Automatically getting the number of classes in the dataset, this in turn acts like the name of the identities in the trained model.
num_classes = len(training_set.class_indices)


# Names of the different classes
class_labels = ['Chris Evans', 'Chris Hemsworth', 'Mark Ruffalo', 'Robert Downey Jr', 'Scarlett Johansson']

# Verify our generator by plotting a few faces and printing corresponding labels
img, label = training_set.__next__()
i = random.randint(0, (img.shape[0])-1)
image = img[i]
labl = class_labels[label[i].argmax()]
plt.imshow(image[:,:,0], cmap='summer')
plt.title(labl)
plt.show()


# Count of files in the training data
num_train_imgs = 0
for root, dirs, files in os.walk(train_data):
    num_train_imgs += len(files)


# Count of files in the test data  
num_test_imgs = 0
for root, dirs, files in os.walk(test_data):
    num_test_imgs += len(files)


# Count of files in the validation data
num_val_imgs = 0
for root, dirs, files in os.walk(val_data):
    num_val_imgs += len(files)


# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=[image_width, image_height, 3]))

# Step 2 - first Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))

# Step 4 - Second Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 5 - Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))

# Step 6 - Second Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 7 - Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'))

# Step 8 - Second Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 9 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 10 - Full Connection
cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))
cnn.add(Dropout(0.5))

# Step 11 - Output Layer
cnn.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
model = cnn.fit(x = training_set, steps_per_epoch=num_train_imgs//batch_size, validation_data = val_set, epochs = 25, validation_steps=num_val_imgs//batch_size, batch_size=batch_size)

# Saving the trained model
cnn.save('identification-model.h5')

# plot the training and validation loss at each epoch
loss = model.history['loss']
val_loss = model.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot the training and validation accuracy at each epoch
acc = model.history['accuracy']
val_acc = model.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


test_loss, test_acc = cnn.evaluate(test_set)

# Use the trained model to predict on the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(test_data, target_size=(image_width, image_height), batch_size=batch_size, class_mode='categorical', shuffle=False)
predictions = cnn.predict(test_set, verbose=1)


# Get the predicted classes and true classes
pred_classes = np.argmax(predictions, axis=1)
true_classes = test_set.classes

# Generate a classification report and confusion matrix
report = classification_report(true_classes, pred_classes, target_names=class_labels)
confusion_mtx = confusion_matrix(true_classes, pred_classes)

# Print the classification report and confusion matrix
print("Classification Report:\n", report)
print("Confusion Matrix:\n", confusion_mtx)


# Get the predicted labels for the test set
test_set_preds = np.argmax(cnn.predict(test_set), axis=1)

# Get the true labels for the test set
test_set_trues = test_set.classes

# Generate the confusion matrix
cm = confusion_matrix(test_set_trues, test_set_preds)

# Define a function to plot the confusion matrix
def plot_confusion_matrix(cm, classes):

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # Plot the confusion matrix values inside the graphical matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Plot the confusion matrix
plot_confusion_matrix(cm, classes=class_labels)
plt.show()


# Compute macro-averaged precision, recall, and F1 score
macro_precision, macro_recall, macro_f1_score, _ = precision_recall_fscore_support(true_classes, pred_classes, average='macro')

# Compute micro-averaged precision, recall, and F1 score
micro_precision, micro_recall, micro_f1_score, _ = precision_recall_fscore_support(true_classes, pred_classes, average='micro')

# Print the evaluation metrics
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)
print('Macro-averaged Precision:', macro_precision)
print('Macro-averaged Recall:', macro_recall)
print('Macro-averaged F1 Score:', macro_f1_score)
print('Micro-averaged Precision:', micro_precision)
print('Micro-averaged Recall:', micro_recall)
print('Micro-averaged F1 Score:', micro_f1_score)