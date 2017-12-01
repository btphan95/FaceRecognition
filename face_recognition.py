# # Sample Face Recognition System

from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import os
import cv2
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from utils import *
from inception import *

# get_ipython().magic('matplotlib inline')
# get_ipython().magic('load_ext autoreload')
# get_ipython().magic('autoreload 2')

np.set_printoptions(threshold=np.nan)


# This network uses 96x96 dimensional RGB images as its input. Specifically, inputs a face image (or batch of $m$ face images) as a tensor of shape $(m, n_C, n_H, n_W) = (m, 3, 96, 96)$ 
# It outputs a matrix of shape $(m, 128)$ that encodes each input face image into a 128-dimensional vector
 
# Create the model for face images based on the Inception model

FRmodel = faceRecogModel(input_shape=(3, 96, 96))
print("created faceRecogModel")
### The Triplet Loss Function ###
def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, not needed in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(anchor - positive),axis=None)
    # Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(anchor - negative),axis=None)
    # Subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist,neg_dist),alpha)
    # Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss,0.0))
    
    return loss

print("loading weights")
### Loading the trained model that is exported from an OpenFace torch model
####################################################################################################################################
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)
print("loaded weights")

# Run the following code to build the database (represented as a python dictionary). This database maps each person's name to a 128-dimensional encoding of their face.

database = {}
database["binh"] = img_to_encoding("binh.jpg", FRmodel)
database["binh1"] = img_to_encoding("binh1.jpg", FRmodel)
database["binh2"] = img_to_encoding("binh2.jpg", FRmodel)
database["binh3"] = img_to_encoding("binh3.jpg", FRmodel)
print("created database")
### Face Recognition
def who_is_it(image_path, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    
    ## Compute the target "encoding" for the image
    encoding = img_to_encoding(image_path,model)
    
    ## Find the closest encoding
    
    # Initialize "min_dist" to a large value
    min_dist = 100
    
    # Loop over the database dictionary's names and encodings
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database
        dist = np.linalg.norm(db_enc - encoding)
        print(name, " has distance ", dist)
        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > 0.7:
        print("Not in the database. The distance is " + str(min_dist))
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))

        
    return min_dist, identity


#####recognition test
print("recognizing face")
who_is_it("selfie.jpg", database, FRmodel)