#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install --upgrade tensorflow==2.3.1


# In[2]:


#We generate embeddings for the images - embeddings in ResNet are 2048nset of numbers representing the image for this we use ResNet model and use this model to generate the embeddings
import tensorflow
from tensorflow.keras.preprocessing import image  #To parse the images
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input


# In[3]:


#we create our ResNet model by calling ResNet class
#we use weights which are trained on imagenet dataset 
#include_top=False because we will add our own top layer
#input_shape=(224,224,3) we are using the standard size of the images for the input images - it scales down the images of larger size to this standard size
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
#We are not training our model since its already trained on imagenet dataset
model.trainable = False

#we are giving our own top layer using sequential module
#sequential helps us in creating model layers and adding them
model = tensorflow.keras.Sequential([
    model,  #first we are passing our model
    GlobalMaxPooling2D() #then we are passing GlobalMaxPooling2D layer i.e we addedour own top layer
])


# In[4]:


print(model.summary()) #we used max pooling layer to change the output shape


# ##FOR SINGLE IMAGE
# 
# imported image using keras.preprocessing
# 
# used utility function called image.load_img() which is in image and set the size(target())
# 
# we convert image to array using image.img_array() this is 3D image
# 
# we reshap single image because it works on batch of images and not single image so we use np.expand_dims() which is 4d image
# 
# import other tensorflow libraries
# 
# create the model as done above
# 
# we send the expanded image to preprocess_unit which is present in ResNet50
# 
# preprocess_unit - coverts the input we are giving to Resnet model to a correct format. Since ResNet model is trainedd in imagenet data so we want input in that format itself
# 
# Returns Preprocessed numpy.array or a tf.Tensor with type float32. The images are converted from RGB to BGR, then each color channel is zero-centered with respect to the ImageNet dataset, without scaling.
# 
# we give this preprocess_unit to the ResNet model i.e mode.predict(p_u) and apply .flatten() to convert it to 1D image
# 
# we normalize the values i.e lies between 0 to 1, this is done by dividing each value by norm(the entire embedding value)

# In[16]:


import numpy as np
from numpy.linalg import norm


# In[17]:


#creating a function which requires img path and a model
def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224)) #load the image
    img_array = image.img_to_array(img)       #converting image to array
    expanded_img_array = np.expand_dims(img_array, axis=0)   #expanding the dimensions i.e 3D TO 4D
    preprocessed_img = preprocess_input(expanded_img_array) #Explanation is given above
    result = model.predict(preprocessed_img).flatten() #predict and flatten the result
    normalized_result = result / norm(result)  #normalize 

    return normalized_result


# One by one we send our 44k images to this function and get their norm results i.e get the extracted features
# 
# for this we 
# create a list which contains teh filenames of the images that are there in the image folder

# In[18]:


import os
from tqdm import tqdm 
import pickle


# In[12]:


print(os.listdir(("C:/Users/ahmed/Downloads/archive (2)/images"))) #the output will file names


# In[19]:


filenames = []

for file in os.listdir("C:/Users/ahmed/Downloads/archive (2)/images"):
    filenames.append(os.path.join("C:/Users/ahmed/Downloads/archive (2)/images",file))


# In[15]:


print(len(filenames))
print(filenames[0:5])


# For each file we call the extract function and this fnction will rxtract the features of that file/image and return us
# 
# We create feature_list which is 2D, it'll contain 44k lists i.e for each image a list is made. inside each image list there will be 2048 features of that image.  

# In[20]:


feature_list = []

for file in tqdm(filenames):  #tqdm shows you the progress of for loop we use this because this loop takes a lot of time to run i.e almost an hour
    feature_list.append(extract_features(file,model))

#we need tro export the feature_list for future use . We use pickle 
pickle.dump(feature_list,open('embeddings.pkl','wb')) #we are first dumbping the feature list and giving the file name as embeddings and write in binary mode
pickle.dump(filenames,open('filenames.pkl','wb')) #we dump filenames 
#when the code gets completed, these two .pkl files are exported and we can use these files anywhere

