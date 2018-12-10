#TrainHolidayV3.ipynb 
#Created by Noah Walsh, Ben Valois, Rick Djeuhon, Derek Windahl, and Jake Hamilton
#Deep Learning Holiday Image Manipulation
#Trains the Inception V3 classifier with the custom Holiday_images dataset, and saves the weights
#for later use in HolidayImageManipulator.py
#Make sure to have the Holiday_images training and testing folder downloaded and in the correct location, and
#copy and paste this code into an ipython shell in Noah Walsh's Dev Cloud account in the correct environment, details
#laid out in the ReadMe file.
#12/8/18

from __future__ import print_function

from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
import scipy

from keras.applications import inception_v3
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
# dimensions of our images.

img_width, img_height = 150, 150



train_data_dir = 'Holiday_images/train'

validation_data_dir = 'Holiday_images/val' #break into train and val

nb_train_samples = 2013

nb_validation_samples = 481 #Usually set val to 0 to maximise training dataset

epochs = 2

batch_size = 10

#Set input image data format
if K.image_data_format() == 'channels_first':

    input_shape = (3, img_width, img_height)

else:

    input_shape = (img_width, img_height, 3)

# Shows Loss and Accuracy correctly
K.set_learning_phase(1)

# Build the InceptionV3 network with our placeholder.
# The model will be loaded with our pre-trained holiday weights.
model = inception_v3.InceptionV3(weights=None, #If no pre-trained holiday weights already saved, use weights = None
                                 include_top=True, input_shape=input_shape) #include fully connected layer for training, fix input array shape to same shape as training images
model.summary()

model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy']) #Compile the model

train_datagen = ImageDataGenerator( #Prepare and rescale training data

    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)

#Training images are labeled by which folder they are in, so for 1000 folders, there are potentially 1000 labels.
#Files in folder are the x-training data and the folder name is the y-classification label.
train_generator = train_datagen.flow_from_directory( 

    train_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size)

test_datagen = ImageDataGenerator(rescale=1. / 255) 

validation_generator = test_datagen.flow_from_directory(

    validation_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size)

#Train model on labelled holiday images.
model.fit_generator(

    train_generator,

    steps_per_epoch=nb_train_samples // batch_size,

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=nb_validation_samples // batch_size)

model.save_weights('holiday_weights.h5') #save trained holiday weights for image manipulation
