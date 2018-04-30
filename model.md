

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
%matplotlib inline
import os
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import pandas as pd
import datetime
import matplotlib.image as mpimg
```

    Using TensorFlow backend.



```python
#load data and name columns
data_csv = pd.read_csv("data/driving_log.csv",sep=",")
data_csv.columns=["center_image","right_image","left_image","steering_angle","throttle","break","speed"]
```


```python
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
```


```python
#add new columns to dataframe
data_csv["train_session"]=training_session
data_csv["time_diff"]=time_diff_list
data_csv=data_csv.loc[data_csv["train_session"]>=23]
```


```python
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
```


```python
data_csv
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>center_image</th>
      <th>right_image</th>
      <th>left_image</th>
      <th>steering_angle</th>
      <th>throttle</th>
      <th>break</th>
      <th>speed</th>
      <th>train_session</th>
      <th>time_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6478</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15021</td>
      <td>23</td>
      <td>1.562</td>
    </tr>
    <tr>
      <th>6479</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15021</td>
      <td>23</td>
      <td>0.105</td>
    </tr>
    <tr>
      <th>6480</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15021</td>
      <td>23</td>
      <td>0.115</td>
    </tr>
    <tr>
      <th>6481</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15023</td>
      <td>23</td>
      <td>0.115</td>
    </tr>
    <tr>
      <th>6482</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15021</td>
      <td>23</td>
      <td>0.116</td>
    </tr>
    <tr>
      <th>6483</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15020</td>
      <td>23</td>
      <td>0.101</td>
    </tr>
    <tr>
      <th>6484</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15021</td>
      <td>23</td>
      <td>0.101</td>
    </tr>
    <tr>
      <th>6485</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>-0.113858</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.14808</td>
      <td>23</td>
      <td>0.116</td>
    </tr>
    <tr>
      <th>6486</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>-0.084473</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.14986</td>
      <td>23</td>
      <td>0.100</td>
    </tr>
    <tr>
      <th>6487</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>-0.071879</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15040</td>
      <td>23</td>
      <td>0.102</td>
    </tr>
    <tr>
      <th>6488</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>-0.071879</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15011</td>
      <td>23</td>
      <td>0.115</td>
    </tr>
    <tr>
      <th>6489</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15019</td>
      <td>23</td>
      <td>0.100</td>
    </tr>
    <tr>
      <th>6490</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15018</td>
      <td>23</td>
      <td>0.115</td>
    </tr>
    <tr>
      <th>6491</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15019</td>
      <td>23</td>
      <td>0.100</td>
    </tr>
    <tr>
      <th>6492</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.021504</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15011</td>
      <td>23</td>
      <td>0.117</td>
    </tr>
    <tr>
      <th>6493</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.021504</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15014</td>
      <td>23</td>
      <td>0.101</td>
    </tr>
    <tr>
      <th>6494</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.021504</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15018</td>
      <td>23</td>
      <td>0.118</td>
    </tr>
    <tr>
      <th>6495</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.021504</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15018</td>
      <td>23</td>
      <td>0.115</td>
    </tr>
    <tr>
      <th>6496</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.021504</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15020</td>
      <td>23</td>
      <td>0.101</td>
    </tr>
    <tr>
      <th>6497</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.008910</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15022</td>
      <td>23</td>
      <td>0.100</td>
    </tr>
    <tr>
      <th>6498</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15021</td>
      <td>23</td>
      <td>0.115</td>
    </tr>
    <tr>
      <th>6499</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15018</td>
      <td>23</td>
      <td>0.117</td>
    </tr>
    <tr>
      <th>6500</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15021</td>
      <td>23</td>
      <td>0.101</td>
    </tr>
    <tr>
      <th>6501</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15020</td>
      <td>23</td>
      <td>0.101</td>
    </tr>
    <tr>
      <th>6502</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15022</td>
      <td>23</td>
      <td>0.100</td>
    </tr>
    <tr>
      <th>6503</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15017</td>
      <td>23</td>
      <td>0.114</td>
    </tr>
    <tr>
      <th>6504</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15022</td>
      <td>23</td>
      <td>0.104</td>
    </tr>
    <tr>
      <th>6505</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>-0.155837</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.14460</td>
      <td>23</td>
      <td>0.114</td>
    </tr>
    <tr>
      <th>6506</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>-0.155837</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.14761</td>
      <td>23</td>
      <td>0.101</td>
    </tr>
    <tr>
      <th>6507</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.14947</td>
      <td>23</td>
      <td>0.114</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>40952</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>-0.254094</td>
      <td>0.787051</td>
      <td>0.0</td>
      <td>30.13533</td>
      <td>499</td>
      <td>0.111</td>
    </tr>
    <tr>
      <th>40953</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>-0.229035</td>
      <td>0.787051</td>
      <td>0.0</td>
      <td>30.13628</td>
      <td>499</td>
      <td>0.101</td>
    </tr>
    <tr>
      <th>40954</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>-0.074500</td>
      <td>0.798309</td>
      <td>0.0</td>
      <td>30.14070</td>
      <td>499</td>
      <td>0.115</td>
    </tr>
    <tr>
      <th>40955</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>-0.061971</td>
      <td>0.798309</td>
      <td>0.0</td>
      <td>30.14022</td>
      <td>499</td>
      <td>0.106</td>
    </tr>
    <tr>
      <th>40956</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.798309</td>
      <td>0.0</td>
      <td>30.13924</td>
      <td>499</td>
      <td>0.111</td>
    </tr>
    <tr>
      <th>40957</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.798309</td>
      <td>0.0</td>
      <td>30.13925</td>
      <td>499</td>
      <td>0.109</td>
    </tr>
    <tr>
      <th>40958</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>-0.045264</td>
      <td>0.798309</td>
      <td>0.0</td>
      <td>30.13879</td>
      <td>499</td>
      <td>0.115</td>
    </tr>
    <tr>
      <th>40959</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>-0.095383</td>
      <td>0.798309</td>
      <td>0.0</td>
      <td>30.13771</td>
      <td>499</td>
      <td>0.111</td>
    </tr>
    <tr>
      <th>40960</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>-0.145503</td>
      <td>0.798309</td>
      <td>0.0</td>
      <td>30.13643</td>
      <td>499</td>
      <td>0.104</td>
    </tr>
    <tr>
      <th>40961</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>-0.158032</td>
      <td>0.798309</td>
      <td>0.0</td>
      <td>30.13722</td>
      <td>499</td>
      <td>0.111</td>
    </tr>
    <tr>
      <th>40962</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>-0.170562</td>
      <td>0.791362</td>
      <td>0.0</td>
      <td>30.13576</td>
      <td>499</td>
      <td>0.100</td>
    </tr>
    <tr>
      <th>40963</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>-0.183092</td>
      <td>0.778430</td>
      <td>0.0</td>
      <td>30.13265</td>
      <td>499</td>
      <td>0.103</td>
    </tr>
    <tr>
      <th>40964</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>-0.183092</td>
      <td>0.778430</td>
      <td>0.0</td>
      <td>30.13305</td>
      <td>499</td>
      <td>0.116</td>
    </tr>
    <tr>
      <th>40965</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>-0.183092</td>
      <td>0.778430</td>
      <td>0.0</td>
      <td>30.13331</td>
      <td>499</td>
      <td>0.102</td>
    </tr>
    <tr>
      <th>40966</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.752566</td>
      <td>0.0</td>
      <td>30.13353</td>
      <td>499</td>
      <td>0.112</td>
    </tr>
    <tr>
      <th>40967</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>29.88691</td>
      <td>499</td>
      <td>0.100</td>
    </tr>
    <tr>
      <th>40968</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>29.58737</td>
      <td>499</td>
      <td>0.101</td>
    </tr>
    <tr>
      <th>40969</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>-0.329273</td>
      <td>0.028380</td>
      <td>0.0</td>
      <td>29.19674</td>
      <td>499</td>
      <td>0.115</td>
    </tr>
    <tr>
      <th>40970</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>-0.696814</td>
      <td>0.153389</td>
      <td>0.0</td>
      <td>28.65348</td>
      <td>499</td>
      <td>0.101</td>
    </tr>
    <tr>
      <th>40971</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>28.29328</td>
      <td>499</td>
      <td>0.118</td>
    </tr>
    <tr>
      <th>40972</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>28.00284</td>
      <td>499</td>
      <td>0.100</td>
    </tr>
    <tr>
      <th>40973</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>27.72089</td>
      <td>499</td>
      <td>0.115</td>
    </tr>
    <tr>
      <th>40974</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>27.38712</td>
      <td>499</td>
      <td>0.116</td>
    </tr>
    <tr>
      <th>40975</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>27.11213</td>
      <td>499</td>
      <td>0.100</td>
    </tr>
    <tr>
      <th>40976</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>26.78576</td>
      <td>499</td>
      <td>0.116</td>
    </tr>
    <tr>
      <th>40977</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>26.51681</td>
      <td>499</td>
      <td>0.102</td>
    </tr>
    <tr>
      <th>40978</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>26.25056</td>
      <td>499</td>
      <td>0.100</td>
    </tr>
    <tr>
      <th>40979</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>25.93459</td>
      <td>499</td>
      <td>0.116</td>
    </tr>
    <tr>
      <th>40980</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>25.62242</td>
      <td>499</td>
      <td>0.115</td>
    </tr>
    <tr>
      <th>40981</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>25.36516</td>
      <td>499</td>
      <td>0.103</td>
    </tr>
  </tbody>
</table>
<p>34504 rows × 9 columns</p>
</div>




```python
# basic plot
print(min(data_csv["steering_angle"]))
print(max(data_csv["steering_angle"]))
plt.figure(figsize=(20,10))
plt.boxplot(data_csv["steering_angle"])
```


```python
for unique_session in np.unique(data_csv["train_session"]):
    plt.figure(figsize=(20,10))
    #plt.plot(data_csv.loc[data_csv["train_session"]==unique_session]["steering_angle_rm_3"])
    #plt.plot(data_csv.loc[data_csv["train_session"]==unique_session]["steering_angle_rm_5"])
    plt.plot(data_csv.loc[data_csv["train_session"]==unique_session]["steering_angle"])
    plt.show()
```


```python
plt.figure(figsize=(20,10))
plt.plot(data_csv["steering_angle"])
```


```python
def scaling(img,scale_x,scale_y):
    return cv2.resize(img,None,fx=scale_x, fy=scale_y, interpolation = cv2.INTER_LINEAR)
```


```python
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
```


```python
#split training set into training and validation set
from sklearn.cross_validation import train_test_split

y_train, y_validation = train_test_split(data_csv, test_size=0.2, random_state=88)
```


```python
from sklearn.utils import shuffle
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for index, batch_sample in batch_samples.iterrows():
                name = batch_sample[0]
                center_image = min_max_normalization(scaling(mpimg.imread(name),0.5,0.5),0,255)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                images.append(np.fliplr(center_image))
                angles.append(-center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(y_train, batch_size=32)
validation_generator = generator(y_validation, batch_size=32)
```


```python
model = Sequential()
# TODO: Re-construct the network and add a convolutional layer before the flatten layer.
# Create the Sequential model
model.add(Convolution2D(10, 3, 3, border_mode='valid', input_shape=(80, 160, 3)))
model.add(MaxPooling2D((2,2)))

model.add(Activation('relu'))

model.add(Convolution2D(20, 3, 3, border_mode='valid'))
model.add(MaxPooling2D((2,2)))
model.add(Activation('relu'))

model.add(Convolution2D(30, 3, 3, border_mode='valid'))
model.add(MaxPooling2D((2,2)))
model.add(Activation('relu'))

# 1st Layer - Add a flatten layer
model.add(Flatten())

model.add(Dense(800))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# 2nd Layer - Add a fully connected layer
model.add(Dense(200))


# 3rd Layer - Add a ReLU activation layer
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100))

# 3rd Layer - Add a ReLU activation layer
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64))

# 3rd Layer - Add a ReLU activation layer
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32))

# 3rd Layer - Add a ReLU activation layer
model.add(Activation('relu'))

model.add(Dense(16))

# 3rd Layer - Add a ReLU activation layer
model.add(Activation('relu'))
# 4th Layer - Add a fully connected layer
model.add(Dense(1))

# 5th Layer - Add a ReLU activation layer
#model.add(Activation('softmax'))
```


```python

model.compile('adam', 'mean_squared_error', ['mae'])
#model.fit(data_set_features_augmented, y_train, batch_size=128, nb_epoch=5, validation_split=0.2)
model.fit_generator(train_generator, samples_per_epoch= len(y_train)*2, validation_data=validation_generator, nb_val_samples=len(y_validation)*2, nb_epoch=1)
```

    Epoch 1/1
     7552/55206 [===>..........................] - ETA: 547s - loss: 0.0201 - mean_absolute_error: 0.0802


```python
model.save("model.h5")
```


```python
del model

```


```python
float(model.predict(X_normalized[4][None, :, :, :]))
```


```python
images=[]
angles=[]
for index, batch_sample in data_csv[0:10].iterrows():
                name = batch_sample[0]
                print(batch_sample[0])
                center_image = min_max_normalization(scaling(mpimg.imread(name),0.5,0.5),0,255)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
```

    center_image      /home/pepar/Python-Projects/Self_driving_car/C...
    right_image        /home/pepar/Python-Projects/Self_driving_car/...
    left_image         /home/pepar/Python-Projects/Self_driving_car/...
    steering_angle                                                    0
    throttle                                                   0.841635
    break                                                             0
    speed                                                       30.1502
    train_session                                                    23
    time_diff                                                     1.562
    Name: 6478, dtype: object
    center_image      /home/pepar/Python-Projects/Self_driving_car/C...
    right_image        /home/pepar/Python-Projects/Self_driving_car/...
    left_image         /home/pepar/Python-Projects/Self_driving_car/...
    steering_angle                                                    0
    throttle                                                   0.841635
    break                                                             0
    speed                                                       30.1502
    train_session                                                    23
    time_diff                                                     0.105
    Name: 6479, dtype: object
    center_image      /home/pepar/Python-Projects/Self_driving_car/C...
    right_image        /home/pepar/Python-Projects/Self_driving_car/...
    left_image         /home/pepar/Python-Projects/Self_driving_car/...
    steering_angle                                                    0
    throttle                                                   0.841635
    break                                                             0
    speed                                                       30.1502
    train_session                                                    23
    time_diff                                                     0.115
    Name: 6480, dtype: object
    center_image      /home/pepar/Python-Projects/Self_driving_car/C...
    right_image        /home/pepar/Python-Projects/Self_driving_car/...
    left_image         /home/pepar/Python-Projects/Self_driving_car/...
    steering_angle                                                    0
    throttle                                                   0.841635
    break                                                             0
    speed                                                       30.1502
    train_session                                                    23
    time_diff                                                     0.115
    Name: 6481, dtype: object
    center_image      /home/pepar/Python-Projects/Self_driving_car/C...
    right_image        /home/pepar/Python-Projects/Self_driving_car/...
    left_image         /home/pepar/Python-Projects/Self_driving_car/...
    steering_angle                                                    0
    throttle                                                   0.841635
    break                                                             0
    speed                                                       30.1502
    train_session                                                    23
    time_diff                                                     0.116
    Name: 6482, dtype: object
    center_image      /home/pepar/Python-Projects/Self_driving_car/C...
    right_image        /home/pepar/Python-Projects/Self_driving_car/...
    left_image         /home/pepar/Python-Projects/Self_driving_car/...
    steering_angle                                                    0
    throttle                                                   0.841635
    break                                                             0
    speed                                                       30.1502
    train_session                                                    23
    time_diff                                                     0.101
    Name: 6483, dtype: object
    center_image      /home/pepar/Python-Projects/Self_driving_car/C...
    right_image        /home/pepar/Python-Projects/Self_driving_car/...
    left_image         /home/pepar/Python-Projects/Self_driving_car/...
    steering_angle                                                    0
    throttle                                                   0.841635
    break                                                             0
    speed                                                       30.1502
    train_session                                                    23
    time_diff                                                     0.101
    Name: 6484, dtype: object
    center_image      /home/pepar/Python-Projects/Self_driving_car/C...
    right_image        /home/pepar/Python-Projects/Self_driving_car/...
    left_image         /home/pepar/Python-Projects/Self_driving_car/...
    steering_angle                                            -0.113858
    throttle                                                   0.841635
    break                                                             0
    speed                                                       30.1481
    train_session                                                    23
    time_diff                                                     0.116
    Name: 6485, dtype: object
    center_image      /home/pepar/Python-Projects/Self_driving_car/C...
    right_image        /home/pepar/Python-Projects/Self_driving_car/...
    left_image         /home/pepar/Python-Projects/Self_driving_car/...
    steering_angle                                           -0.0844727
    throttle                                                   0.841635
    break                                                             0
    speed                                                       30.1499
    train_session                                                    23
    time_diff                                                       0.1
    Name: 6486, dtype: object
    center_image      /home/pepar/Python-Projects/Self_driving_car/C...
    right_image        /home/pepar/Python-Projects/Self_driving_car/...
    left_image         /home/pepar/Python-Projects/Self_driving_car/...
    steering_angle                                           -0.0718789
    throttle                                                   0.841635
    break                                                             0
    speed                                                       30.1504
    train_session                                                    23
    time_diff                                                     0.102
    Name: 6487, dtype: object



```python
data_csv[0:10]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>center_image</th>
      <th>right_image</th>
      <th>left_image</th>
      <th>steering_angle</th>
      <th>throttle</th>
      <th>break</th>
      <th>speed</th>
      <th>train_session</th>
      <th>time_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6478</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15021</td>
      <td>23</td>
      <td>1.562</td>
    </tr>
    <tr>
      <th>6479</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15021</td>
      <td>23</td>
      <td>0.105</td>
    </tr>
    <tr>
      <th>6480</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15021</td>
      <td>23</td>
      <td>0.115</td>
    </tr>
    <tr>
      <th>6481</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15023</td>
      <td>23</td>
      <td>0.115</td>
    </tr>
    <tr>
      <th>6482</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15021</td>
      <td>23</td>
      <td>0.116</td>
    </tr>
    <tr>
      <th>6483</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15020</td>
      <td>23</td>
      <td>0.101</td>
    </tr>
    <tr>
      <th>6484</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>0.000000</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15021</td>
      <td>23</td>
      <td>0.101</td>
    </tr>
    <tr>
      <th>6485</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>-0.113858</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.14808</td>
      <td>23</td>
      <td>0.116</td>
    </tr>
    <tr>
      <th>6486</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>-0.084473</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.14986</td>
      <td>23</td>
      <td>0.100</td>
    </tr>
    <tr>
      <th>6487</th>
      <td>/home/pepar/Python-Projects/Self_driving_car/C...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>/home/pepar/Python-Projects/Self_driving_car/...</td>
      <td>-0.071879</td>
      <td>0.841635</td>
      <td>0.0</td>
      <td>30.15040</td>
      <td>23</td>
      <td>0.102</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
