#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
import pathlib


# In[4]:


folders = tf.io.gfile.glob("data")
img = tf.io.gfile.glob("data\*\*.jpg")


# In[5]:


len(img)


# In[6]:


#import PIL
#PIL.Image.open(img[-1])


# In[7]:


img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  folders[0],
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  )

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  folders[0],
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  )

from tensorflow.keras import layers
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
from tensorflow.python.keras import regularizers
num_classes = 2

model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(32, 3, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu',kernel_regularizer=regularizers.l2(0.001)),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu',kernel_regularizer=regularizers.l2(0.001)),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu',kernel_regularizer=regularizers.l2(0.001)),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation='relu',kernel_regularizer=regularizers.l2(0.001)),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.001)),
  layers.Dropout(0.7),
  layers.Dense(num_classes)
])


# In[16]:


class_names = train_ds.class_names
print(class_names)


# In[8]:


model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])


# In[9]:


model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=5
)


# In[11]:


model.evaluate(val_ds, steps=None, verbose=1)


# In[18]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




