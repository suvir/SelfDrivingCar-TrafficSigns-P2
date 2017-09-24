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
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](TODO : Add link to github repo)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to normalize the data. This was done using the method prescribed in the lectures for RGB images - **(pixel - 128)/128** .Normalization results in images have zero mean and equal variance. 

Here is an image after normalization:

![alt text][image1]

As a next step, I converted the images to YUV images. This is similar to preprocessing described in a [published baseline model for this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).

Here is the above image after conversion to YUV:

![alt text][image1]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Layer 1 : Convolution 5x5     	| 1x1 stride, valid padding, output = 28x28x10 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  output = 14x14x10 				|
| Layer 2 : Convolution 5x5	    | 1x1 stride, valid padding, output = 10x10x20   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  output = 5x5x20 				|
| Flatten	      	| output = 500 				|
| Layer 3 : Fully connected	| output = 500        									|
| Dropout				| prob = 0.75        									|
| Layer 4 : Fully connected		| output = 200        									|
| Dropout				| prob = 0.75        									|
| Layer 5 : Fully connected(logits)	| output = 43        									|

 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer with batch size = 128.

I trained the model for 50 epochs using a learning rate of 1e-3.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.6%
* validation set accuracy of 94.4%
* test set accuracy of 92.2%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
   * [LeNet-5 architecture](http://yann.lecun.com/exdb/lenet/) was a great starting point
   * Decent performance out of the box @ 89% validation accuracy.  
   * The model was not too complicated and gave me room to experiment with ideas from the lectures. 
   * It was not as computationally intensive as a very large network. I was able to train quickly to validate ideas. 

* What were some problems with the initial architecture?
   *  Did not generalize to new images -  Not performing as well as training data on validation data or new images from the web.


* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
   * Adjusted depth of both convolutional layers. 
   * Adjusted size of the fully connected layers.
   * Added dropout 
   * Added L2 regularization
* Which parameters were tuned? How were they adjusted and why?
   * Number of epochs was increased to ensure that model converges.
   * Regularization factor : tuned to 1e-6
   * Dropout probabilitiy : Set to 0.75 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
   * Convolutional layers work well with this type of problem. Successive convolutional layers can capture increasingly complex details of the images.
   * Dropout layer helps reduce the extent of overfitting.
   * Penalizing the network based on L2 regularization also helps reduce overfitting. 
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The image for 30 km/h speed limit may be difficult to classify because it is captured at an angle. The plane of the sign is not directly exactly parallel to that of the camera.

The STOP image has been captured from beneath the sign. This could make it difficult to classify if all the images in training data are captured from a perfectly straight angle.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 