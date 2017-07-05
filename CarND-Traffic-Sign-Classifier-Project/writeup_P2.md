#**Traffic Sign Recognition** 
**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_images/roundturn.jpeg "Traffic Sign 1"
[image5]: ./test_images/noentry.jpeg "Traffic Sign 2"
[image6]: ./test_images/nopassing.jpeg "Traffic Sign 3"
[image7]: ./test_images/speed50.jpeg "Traffic Sign 4"
[image8]: ./test_images/speed30.jpeg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  


You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the vanilla python and numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the color of the sign does not convey additional message. Also, as [a Convnet paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) suggests, colorless images yield better results

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because normalized data can lead to faster training. Especially for activation functions like relu or sigmoid, unnormalized data can lead to training failure or very slow training speed.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer     		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5*5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	2*2     	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5*5	    | 1X1 stride, valid padding, outputs 10X10X256  |
| RELU					|												|
| Max pooling	2*2      	| 2x2 stride,  outputs 5x5x256 				|
| flatten		| outputs 6400        									|
| Fully connected		| outputs 1024        									|
| Fully connected		| outputs 128        									|
| output layer		| outputs 43        									|
| Softmax				|         									|

 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer, as Adam handles momentum very well. I used a batch size of 256, and 10 epochs. The learning rate is 0.001

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.949
* test set accuracy of 0.932

An iterative approach was chosen:
I first implemented the LeNet archtecture, but it did not perform well because the network was too small to extract sufficient features. Inspired by the [Nvidia paper](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), I chose parameters accordingly. Also, I implemented both high level and low level functions for building the model with tf.layers and tf.nn packages.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The more training data, the better performance of the model. Due to the number of images in the training data is vastly different, some images are pretty difficult to classify.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Roundabout      		| Dangerous curve   									| 
| No entry    			| Turn left ahead 										|
| Stop					| Stop											|
| 50 km/h	      		| Round about					 				|
| 30 km/h			| 30 km/h      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This contradicts to the accuracy on the test set. I think the reasons are twofold. First, the data trained in the model varies a lot for different classes; second, some images give too poor quality to recognize even for a human being like me. The image resize technique might play a role; However, I am not sure about that. I will test more techniques in future to fine tune the model.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located before Step 4 of the Ipython notebook.

The model predicts two images correctly. The stop sign and 30 km/h, with 0.84 and 0.93 probability. For the rest three, the prediction is wrong. The interesting one is the second image. The model is 64% sure of the predicted value, however it is still wrong. I think processed image quality might be responsible. The image preprocessing will be one of my activities in future fine tuning.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .09         			| Dangerous curve(image 1)   									| 
| .64     				| Turn left ahead(image 2) 										|
| .84					| Stop(image 3)										|
| .21	      			| Round about(image 4)				 				|
| .93				    | 30 km/h(image 5)     							|

 

### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I chose the last image in the downloaded images to visualize the neural network feature extraction. The chosen image contains the sign 30 km/h. The Conv layer 1 seems to find the sign in the middle of the image and roughly get the shape of 30. 


