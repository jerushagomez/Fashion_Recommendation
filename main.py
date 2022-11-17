#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import os
from PIL import Image  #to display image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2


# In[2]:


feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))


# In[3]:


model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


# In[4]:


st.title('Fashion Product Recommender System')


# In[5]:


#function to save the file
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('C:/Users/ahmed/Downloads/archive (2)/uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


# In[6]:


#creating a function for feature extraction
def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


# In[7]:


#creating a function for recommend
def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


# In[9]:


uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # feature extract
        features = feature_extraction(os.path.join("C:/Users/ahmed/Downloads/archive (2)/uploads",uploaded_file.name),model)
        #st.text(features)
        # recommendention
        indices = recommend(features,feature_list)
        # display image
        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            imagee = cv2.imread(filenames[indices[0][0]])
            cv2.imshow('Image', imagee)
            st.image(imagee)
        with col2:
            imagee = cv2.imread(filenames[indices[0][1]])
            cv2.imshow('Image', imagee)
            st.image(imagee)
        with col3:
            imagee = cv2.imread(filenames[indices[0][2]])
            cv2.imshow('Image', imagee)
            st.image(imagee)
        with col4:
            imagee = cv2.imread(filenames[indices[0][3]])
            cv2.imshow('Image', imagee)
            st.image(imagee)
        with col5:
            imagee = cv2.imread(filenames[indices[0][4]])
            cv2.imshow('Image', imagee)
            st.image(imagee)
    else:
        st.header("Some error occured in file upload")


# #Run the below commands in the command prompt,
# 
#   #jupyter nbconvert --to script main.ipynb     # convert jupyter notebook to script   
#   
#   #awk '!/ipython/' main.py > temp.py && move temp.py app.py && del main.py     #remove ipython widgets and create app.py
#   
#   #streamlit run app.py
