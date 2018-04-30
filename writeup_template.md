**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model, as well as the model.ipynb since I actually did the work in Jupyter
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* run1.mp4 showing the trained model results

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. I actually did the work in 

### Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 10, 20 and 30 with 2x2 max pooling between each convolution and RELU activation

There is a flatten layer followed by 8 Fully connected layers with RELU activation.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers between fully connected layers 1-6 of 50%. 

The model was trained and validated on separate datasets.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I used a combination of center lane driving, recovering from the left and right sides of the road on track 1 and 2 in several different areas on each track.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to experiment from a small model and work my way up. I split my data into a training and validation set using keras model.fit intially but I fianlly reached a point with my collected data that I had to implement a generator. I collected most of my data at the beginning where I drove several laps on each track in different directions where I focussed on driving in the center of the lane and taking the curves smooth. Then I recorded around 300 recovery exaples. I also realized at the very beginning that the 160x320 image could most likely be scaled down and still yeild decent results. I scaled the input images down to 80x160. 
FOr my intial architecture I began with two convolation layers with 10 and 20 3x3 filter size and activation. I had 2 fully connected layer to 128 and from there 1 down to for the regression output. This gave incredibly poor results. The next network I tried was:
* CNN 10 3x3 with 2x2 maxpooling and RELU activation
* CNN 20 3X3 with 2x2 maxpooling, RELU activation and 50% Dropout
* Fully connected densed to 128 with RELU activation and 50% Dropout
* Fully connected densed to 64 with RELU activation and 50% Dropout
* Fully connected densed to 1
This network gave much better results than the first attempt but it had problems dealing with hard corners. The next model I built was much deeper:
* CNN 10 3x3 with 2x2 maxpooling, RELU activation and 10% Dropout
* CNN 20 3X3 with 2x2 maxpooling, RELU activation and 50% Dropout
* Fully connected densed to 500 with RELU activation and 50% Dropout
* Fully connected densed to 200 with RELU activation and 50% Dropout
* Fully connected densed to 100 with RELU activation
* Fully connected densed to 64 with RELU activation
* Fully connected densed to 16 with RELU activation
* Fully connected densed to 1
I had decent success with this model reaching. I was able to drive the complex track but the simple track had problems at the dirt road and a curve that over looked the lake. I recorded lots of training examples to recovery at the dirt road portions and recovery curve that overlooked at the lake. I then tried this model:
* CNN 10 3x3 with 2x2 maxpooling, RELU activation and 10% Dropout
* CNN 20 3X3 with 2x2 maxpooling, RELU activation and 50% Dropout
* CNN 30 3X3 with 2x2 maxpooling, RELU activation and 50% Dropout
* Fully connected densed to 500 with RELU activation and 50% Dropout
* Fully connected densed to 200 with RELU activation and 50% Dropout
* Fully connected densed to 100 with RELU activation 50% Dropout
* Fully connected densed to 64 with RELU activation 50% Dropout
* Fully connected densed to 16 with RELU activation
* Fully connected densed to 1

This network was terrible on the training and validation set. It couldn't even make simple curves even with several training epochs. I realized that I was dropping out too much of the CNNs at the top. I also remebered from previous research that if n layers are needed for a perfect sized model for the data then it would n/p for a mdoel with dropout where p is probability of a node being kept. This was the final model I coded:

* CNN 10 3x3 with 2x2 maxpooling, RELU activation
* CNN 20 3X3 with 2x2 maxpooling, RELU activation
* CNN 30 3X3 with 2x2 maxpooling, RELU activation
* Fully connected densed to 800 with RELU activation and 50% Dropout
* Fully connected densed to 500 with RELU activation and 50% Dropout
* Fully connected densed to 200 with RELU activation and 50% Dropout
* Fully connected densed to 100 with RELU activation 50% Dropout
* Fully connected densed to 64 with RELU activation 50% Dropout
* Fully connected densed to 32 with RELU activation
* Fully connected densed to 16 with RELU activation
* Fully connected densed to 1

I trained this model with 9 training epochs and reached "val_loss: 0.0221 - val_mean_absolute_error: 0.0848". I had to again alter my generator to get this result. It had never occured to me to flip the images and reverse the steering angle sign to increase the number of examples. Once I saw this in the expanded Behavioral Cloning lesson I impmented this in my generator. This gave 55206 training samples and 13802 validation samples.

Once I tested this model in the simulator I could instantly see the difference in stability around curves, recovery and straight portions of the road. It was finally able to make it around the entire track without driving off the road.

Since I started with dropout in each iteration of my architecture I never had an issue with network overfitting the training data.


####2. Final Model Architecture

The final model architecture is the following:

* CNN 10 3x3 with 2x2 maxpooling, RELU activation
* CNN 20 3X3 with 2x2 maxpooling, RELU activation
* CNN 30 3X3 with 2x2 maxpooling, RELU activation
* Fully connected densed to 800 with RELU activation and 50% Dropout
* Fully connected densed to 500 with RELU activation and 50% Dropout
* Fully connected densed to 200 with RELU activation and 50% Dropout
* Fully connected densed to 100 with RELU activation 50% Dropout
* Fully connected densed to 64 with RELU activation 50% Dropout
* Fully connected densed to 32 with RELU activation
* Fully connected densed to 16 with RELU activation
* Fully connected densed to 1

#### 3. Creation of the Training Set & Training Process

I collected most of my data at the beginning where I drove several laps on each track in different directions where I focussed on driving in the center of the lane and taking the curves smooth. Then I recorded around 300 recovery examples between track 1 and 2 where I recovered from the left and right. I never used the left or right images in training my network.

Initially I trained my network with 21,726 examples. I had to record more examples of recovery at certain points and I drove another few laps on track 1 to get 34504 training examples.

Originally I wasn't using a generator to train my network but I had finally hit a point where I needed to. When I read about how to create a generator I also saw in the Udacity lesson that flipping the image is a good way to increase training examples. I also implemented this in my final solution to give a total of 69008 images adn steering angles.

I was happy to see that I never had problems with overfitting since I included dropout in every architecture I tested. In the end I used 9 training epochs since I noticed early on that the more dropout I had slowed the model improvement epoch over epoch. Originally I used 5 epochs to train the previous tested models. I used an adam optimizer so it didn't have to tune the learning rate.
