#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np


# In[2]:


#importing feature list and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))  #rb -read binary
filenames = pickle.load(open('filenames.pkl','rb'))


# In[3]:


print(np.array(feature_list).shape) #for 44k images we have 2048 features for each image


# In[4]:


#importingg test image
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input


# In[5]:


#creating model (we can also import the model)
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


# In[6]:


img = image.load_img('C:/Users/ahmed/Downloads/archive (2)/sample/watch.jpg',target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)


# In[14]:


pip install opencv-python


# In[15]:


#calculate distance between normalized_reuslt and feature_list for this we use k nearest neighbours
from sklearn.neighbors import NearestNeighbors
import cv2


# In[25]:


#we are finding 5 nearest images
#we are using brute algorithm because we have less images 
#we are using euclidean distance to calculate the distance 
neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean') #we gave 6 instead of 5 because the first image is the input image itself
#we give our input data as feature_list
neighbors.fit(feature_list)


# In[26]:


#we find k neighbours of normalized result from feature_list
distances,indices = neighbors.kneighbors([normalized_result])
#returns distances and indices 
#it gives the indices of the matching array i.e image

print(indices)
#Indices ia a 2D array


# In[28]:


#printing the filenames of images of the above indices
for file in indices[0]:
    print(filenames[file])


# In[22]:


#print the images
#we do not use this module if we are converting to website
#from 0th we are extracting 1 to 6 because the first image is same as input since i took the smaple image from the main images folder 
#if the sample image is not from images folder than use incices[0] in for loop
for file in indices[0][1:6]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output',cv2.resize(temp_img,(512,512)))
    cv2.waitKey(0) #we use waitkey so that the screen does not get disappeared in sometime

