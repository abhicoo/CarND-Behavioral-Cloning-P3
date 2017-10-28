#**Behavioral Cloning** 

##Writeup Template

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model_architecture.png "Model Visualization"
[image2]: ./images/center_1.jpg "Center 1"
[image3]: ./images/center_2.jpg "Center 2"
[image4]: ./images/recovery_1.jpg "Recovery 1"
[image5]: ./images/recovery_2.jpg "Recovery 2"
[image6]: ./images/recovery_3.jpg "Recovery 3"
[image7]: ./images/flipped_1.jpg "Flipped 1"

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model uses the same layer as NVIDIA end to end self driving car model.

My model first uses Keras Lambda layer to Normalize the input.

My model consists of 5 convolution layers. 

The first three layers have filter size of 5x5, depths of 24, 36 and 48 respectively and stride of 2.

The last two layers have filter size of 3X3, depths of 64 and 64 espectively and stride of 1.

My model consists of 3 fully connected layers.

The model includes RELU layers to introduce nonlinearity.

The model was trained and validated on both the tracks. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was createad by driving vehicle manually on road. 

Model was first trained on center driving data. 

Images from 3 cameras was use to train the model to stay on the center of the road.

Data Augmentation techniques was used to train the model on different types of data.
The data augmentation technique I used was to flip all the 3 images. It not only helped the model get more training data but to recover from bias toward the training data.

Data was also created to teach model on how to recover when the vehicle went too left or too right. 
To collect this data the vehicle was taken to left or right side of road and then the data was recorded while coming back to center.
The recovery data was collected from both track 1 and track 2.

The center driving data was collected by driving 3-4 laps on track 1.

####1. Solution Design Approach

The key point in the project was to create a good model architecture and do data augmentation.

I used the NVIDIA model to train the network as it already was proven to work nicely. 
The model really worked very nice. 
The vehicle was able to drive both on track 1 and track 2.

Initially sometimes the vehicle was not able to recover back to center after going too left and right. But by using the recovery data this problem was solved.

I split the data into training and validation step and ensured that model was performing good and giving low mean sqaured error on training and validation data.

####2. Final Model Architecture

The final model architecture (model.py lines 25-118) consisted of a 5 convolution layers with the 3 fully conntected layers.

Here is a visualization of the architecture.

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 3-4 laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]
![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how to go back to center if it want too left or right. 
These images show what a recovery looks like:

![alt text][image4]
![alt text][image5]
![alt text][image6]

Then I repeated this process on track two in order to get more data points. I did a 1 lap of driving on track 2. But created more recovering data for track 2 for sharp turns.

To augment the data sat, I also flipped images and angles thinking that this would remove the bias towards the tracking data and also teach model how to drive if in flipped scenario for better generalization. It also helped in providing more training data to model. 
For example, here is an image that has then been flipped:

![alt text][image7]

After the collection process, I had 15000 number of data points. 
Each data point had 3 image center, left, right. The steering angle for left and right images was calculated by adding and subtracing a correction factor of 0.35 from the center steering angle.
All the 3 image where also flipped with there steering angle also flipped by multiplying by -1.
The images where also zero centered and normalized by using Keras Lambda layer.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was 5. Found out by checking the training and validation loss and choosing number where both where low.
I used an adam optimizer so that manually training the learning rate wasn't necessary.
