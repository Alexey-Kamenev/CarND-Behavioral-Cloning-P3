# Behavioral Cloning

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
#### Files Submitted & Code Quality

1. Submission includes all required files and can be used to run the simulator in autonomous mode
My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

3. Submission code is usable and readable
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

#### Model Architecture and Training Strategy

1. An appropriate model architecture has been employed
My model consists of a relatively small convolutional neural network that has 3 convolutional layers, each followed by max pooling laers. Convolutional part (feature extraction) is followed by 2 fully-connected layers. All convolutional and fully-connected layers use ReLU as non-linearity and batch normalization to speed up the training. See lines 115 - 134 in ```model.py``` for more details.
The reason for such relatively simple model is limitations imposed by the data. The simulator data does not vary much so larger networks easily overfit during the training. I tried training larger network (e.g. ```net_medium_bn``` line 137) and got slightly lower validation error, however, the behavior in simulator was worse than of the simple model: too much oscillations during driving. Such behavior is known very well and especially pronounced when training large networks on simulator. Even in case of real-world systems, like drones, network overfitting is a serious issue which affects model quality, that is, how well model controls the vehicle but not model accuracy (the accuracy can still be high). As an example, this effect can be seen in my work project: [Autonomous Drone Navigation with Deep Learning](https://www.youtube.com/watch?v=voVxIGyeqgo).
The data is cropped and normalized in the model using a Keras lambda layer (code lines 176-177). 

2. Attempts to reduce overfitting in the model
As the model has batch normalization layers, using other techniques like dropout is not very beneficial. When experimenting with the network I tried adding dropout after each fully-connected layer but that did not improve overall accuracy or quality of the model even after trainig for longer number of iterations as required in case of dropout.
The model was trained and validated on different data sets to ensure that the model was not overfitting (see generator-related functions like ```train_generator```, ```val_generator``` and ```create_train_val_gen```). I did not use traditional data augmentation techniques like color/contrast/brightnes/etc jitter or rotation and scaling as this does not help much with simulators.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

3. Model parameter tuning
The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 184). I tried other optimizers like SGD with NAG as well as Adadelta but Adam turned out to be the best.

4. Appropriate training data
Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I used data from all 3 cameras.
For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy
1. Solution Design Approach
The overall strategy for deriving a model architecture was to start with a simple architecture and small subset of data and increase model size and add data as needed while testing models in simulator.
From my experience, the biggest problem for creating good model for autonomous navigation (whether car or drone) is that validation set accuracy is not a very good measure of model quality. By "quality" I mean behavior of the vehicle that is controlled by a model. What I often observe is that large model that produce good results on validation or test sets behave less than optimal in real life: the control is "too perfect", for example, the model jerks the vehicle from one side to another. Ideal model would learn how to do an optimal control.

2. Final Model Architecture
The model archetecture and the reasoning behind choosing such architecture have been already discussed in previous sections.

3. Creation of the Training Set & Training Process
I recorded several runs of each of the two tracks. First, I did one full lap for both of the tracks trying to keep the car at the center of the road. I had to use mouse instead of keyboard to enable smoother movements and better steering angle data. Then I recorded recovery segments for both of the tracks where I drive from side to the center. Finally, I repeated the process for each track driving in the opposite direction.

I finally randomly shuffled the data set and put 5% of the data into a validation set. 

Log:
```
Train size: 25920
Val size  : 1344
Epoch 1/10
25920/25920 [==============================] - 342s - loss: 0.1502 - val_loss: 0.0497
Epoch 2/10
25920/25920 [==============================] - 338s - loss: 0.0520 - val_loss: 0.0387
Epoch 3/10
25920/25920 [==============================] - 347s - loss: 0.0443 - val_loss: 0.0329
Epoch 4/10
25920/25920 [==============================] - 346s - loss: 0.0431 - val_loss: 0.0292
Epoch 5/10
25920/25920 [==============================] - 354s - loss: 0.0385 - val_loss: 0.0317
Epoch 6/10
25920/25920 [==============================] - 354s - loss: 0.0376 - val_loss: 0.0326
Epoch 7/10
25920/25920 [==============================] - 359s - loss: 0.0356 - val_loss: 0.0307
Epoch 8/10
25920/25920 [==============================] - 360s - loss: 0.0340 - val_loss: 0.0304
Epoch 9/10
25920/25920 [==============================] - 363s - loss: 0.0326 - val_loss: 0.0252
Epoch 10/10
25920/25920 [==============================] - 345s - loss: 0.0313 - val_loss: 0.0260
```
