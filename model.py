
# coding: utf-8

# In[232]:

import numpy as np
import cv2
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
get_ipython().magic('matplotlib inline')
import os
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import pandas as pd
import datetime
import matplotlib.image as mpimg


# In[241]:

#load data and name columns
data_csv = pd.read_csv("data/driving_log.csv",sep=",")
data_csv.columns=["center_image","right_image","left_image","steering_angle","throttle","break","speed"]


# In[242]:

#create next frame time difference and session count based on time diff
#the session count can be used to associate training sessions and remove poor training sessions if need be
time_diff_list=[0]
training_session=[1]
k=1
for i in range(len(data_csv)-1):
    time_diff=(datetime.datetime.strptime(data_csv["center_image"].iloc[i+1][89:112],"%Y_%m_%d_%H_%M_%S_%f")-
         datetime.datetime.strptime(data_csv["center_image"].iloc[i][89:112],"%Y_%m_%d_%H_%M_%S_%f")).total_seconds()
    
    if  time_diff>0.5:
        k+=1
    time_diff_list.append(time_diff)
    training_session.append(k)
    #print(data_csv["center_image"].iloc[i+1][89:112])


# In[243]:

#add new columns to dataframe
data_csv["train_session"]=training_session
data_csv["time_diff"]=time_diff_list
data_csv=data_csv.loc[data_csv["train_session"]>=22]


# In[244]:

list_rm3=[]
list_rm5=[]
for unique_session in np.unique(data_csv["train_session"]):
    data_rm3=pd.DataFrame(pd.rolling_mean(data_csv.loc[data_csv["train_session"]==unique_session]["steering_angle"],window=3,center=True))
    data_rm5=pd.DataFrame(pd.rolling_mean(data_csv.loc[data_csv["train_session"]==unique_session]["steering_angle"],window=5,center=True))
    data_rm3.columns=["steering_angle_rm_3"]
    data_rm5.columns=["steering_angle_rm_5"]
    data_rm3["steering_angle_rm_3"]=data_rm3["steering_angle_rm_3"].fillna(0)
    data_rm5["steering_angle_rm_5"]=data_rm5["steering_angle_rm_5"].fillna(0)
    list_rm3.extend(list(data_rm3["steering_angle_rm_3"]))
    list_rm5.extend(list(data_rm5["steering_angle_rm_5"]))
data_csv["steering_angle_rm_3"]=list_rm3
data_csv["steering_angle_rm_5"]=list_rm5


# In[245]:

data_csv


# In[246]:

# basic plot
print(min(data_csv["steering_angle"]))
print(max(data_csv["steering_angle"]))
plt.figure(figsize=(20,10))
plt.boxplot(data_csv["steering_angle"])


# In[247]:

for unique_session in np.unique(data_csv["train_session"]):
    plt.figure(figsize=(20,10))
    #plt.plot(data_csv.loc[data_csv["train_session"]==unique_session]["steering_angle_rm_3"])
    #plt.plot(data_csv.loc[data_csv["train_session"]==unique_session]["steering_angle_rm_5"])
    plt.plot(data_csv.loc[data_csv["train_session"]==unique_session]["steering_angle"])
    plt.show()


# In[250]:

plt.figure(figsize=(20,10))
plt.plot(data_csv["steering_angle"])


# In[177]:

def scaling(img,scale_x,scale_y):
    return cv2.resize(img,None,fx=scale_x, fy=scale_y, interpolation = cv2.INTER_LINEAR)


# In[248]:

i=0
for image_path in data_csv["center_image"]:
    image = mpimg.imread(image_path)
    image=scaling(image,160.0/image.shape[1],80.0/image.shape[0])
    plt.imshow(image)
    if i == 0:
        data_set_features_augmented=np.array(image)[np.newaxis,...]
    else:
        data_set_features_augmented=np.concatenate((data_set_features_augmented,np.array(image)[np.newaxis,...]))
    i+=1


# In[249]:

data_set_features_augmented.shape


# In[163]:

#http://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python
def myround(x, base=5):
    return int(base * round(float(x["steering_angle_rm_3"])*100/base))


# In[164]:

data_csv["rm3_bucket"]=data_csv.apply(myround,axis=1)
data_csv["rm3_bucket"]


# In[166]:


from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb.fit(data_csv["rm3_bucket"])
lb.classes_
y_one_hot=lb.transform(data_csv["rm3_bucket"])


# In[173]:

len(lb.classes_)


# In[251]:

#create normalizing function to normailize augmented data set
def min_max_normalization(x,min,max):
    """
    This function takes an n by m array and normalizes each value based on the average of min and max values
    of the RBG scale (min=0 and max=255).
    
    It return an n by m array
    """
    avg_value=(max+min)/2.0
    norm_array = np.zeros(x.shape)+avg_value
    normalized_x= (x-norm_array)/norm_array
    return normalized_x


# In[257]:

X_normalized=min_max_normalization(data_set_features_augmented,0,255)
# TODO: Shuffle the data
from sklearn.utils import shuffle
X_normalized, y_train = shuffle(X_normalized, data_csv["steering_angle"], random_state=88)


# In[266]:

model = Sequential()
# TODO: Re-construct the network and add a convolutional layer before the flatten layer.
# Create the Sequential model
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(80, 160, 3)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

# 1st Layer - Add a flatten layer
model.add(Flatten())

# 2nd Layer - Add a fully connected layer
model.add(Dense(128))

# 3rd Layer - Add a ReLU activation layer
model.add(Activation('relu'))

# 4th Layer - Add a fully connected layer
model.add(Dense(1))

# 5th Layer - Add a ReLU activation layer
#model.add(Activation('softmax'))


# In[271]:


model.compile('adam', 'mean_squared_error', ['mae'])
model.fit(X_normalized, y_train, batch_size=128, nb_epoch=5, validation_split=0.2)


# In[272]:

model.save("model.h5")


# In[265]:

del model


# In[ ]:



