# In[1]:

import tensorflow as tf


# In[2]:


for device in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)


# In[3]:

tf.__version__


# In[4]:


w = 128
h = 128
batch_size = 16 
directory = r"D:\UMD\Courses\673\Project 4\Fish_Dataset\Fish_Dataset"

from tensorflow.keras.preprocessing import image_dataset_from_directory

training_set = image_dataset_from_directory(directory,validation_split = 0.2,subset = "training",seed = 123, label_mode = 'categorical', image_size = (h, w), batch_size = batch_size)
validation_set = image_dataset_from_directory(directory,validation_split = 0.2,subset = "validation",label_mode = 'categorical',seed = 123, image_size = (h, w), batch_size = batch_size)


# In[5]:


AUTOTUNE = tf.data.AUTOTUNE

training_set = training_set.cache().prefetch(buffer_size = AUTOTUNE)
validation_set = validation_set.cache().prefetch(buffer_size = AUTOTUNE)


# In[6]:

classifier = tf.keras.Sequential()

w = 128
h = 128

classes = 9

classifier.add(tf.keras.layers.Conv2D(filters = 64,kernel_size = 3,padding = "same",activation = 'relu',input_shape = (w,h,3)))
classifier.add(tf.keras.layers.Conv2D(filters = 64,kernel_size = 3,padding = "same",activation = 'relu'))

classifier.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2),strides = (2,2)))
classifier.add(tf.keras.layers.Conv2D(filters = 128,kernel_size = 3,padding = "same",activation = 'relu'))
classifier.add(tf.keras.layers.Conv2D(filters = 128,kernel_size = 3,padding = "same",activation = 'relu'))

classifier.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2),strides = (2,2)))
classifier.add(tf.keras.layers.Conv2D(filters = 256,kernel_size = 3,padding = "same",activation = 'relu'))
classifier.add(tf.keras.layers.Conv2D(filters = 256,kernel_size = 3,padding = "same",activation = 'relu'))
classifier.add(tf.keras.layers.Conv2D(filters = 256,kernel_size = 3,padding = "same",activation = 'relu'))

classifier.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2),strides = (2,2)))
classifier.add(tf.keras.layers.Conv2D(filters=512,kernel_size = 3,padding = "same",activation = 'relu'))
classifier.add(tf.keras.layers.Conv2D(filters=512,kernel_size = 3,padding = "same",activation = 'relu'))
classifier.add(tf.keras.layers.Conv2D(filters=512,kernel_size = 3,padding = "same",activation = 'relu'))

classifier.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2),strides = (2,2)))
classifier.add(tf.keras.layers.Conv2D(filters = 512,kernel_size = 3,padding = "same",activation = 'relu'))
classifier.add(tf.keras.layers.Conv2D(filters = 512,kernel_size = 3,padding = "same",activation = 'relu'))
classifier.add(tf.keras.layers.Conv2D(filters = 512,kernel_size = 3,padding = "same",activation = 'relu'))

classifier.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2),strides = (2,2)))
classifier.add(tf.keras.layers.Flatten())
classifier.add(tf.keras.layers.Dense(4090,'relu'))
classifier.add(tf.keras.layers.Dense(4090,'relu'))
classifier.add(tf.keras.layers.Dense(classes,'softmax'))

# In[7]:

classifier.summary()


# In[8]:

    
import tensorboard



# In[9]:
    

classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])



# In[10]:

from datetime import datetime

logdir="/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir)

history = classifier.fit(training_set, validation_data = validation_set, epochs = 1, callbacks = [tensorboard_callback])



