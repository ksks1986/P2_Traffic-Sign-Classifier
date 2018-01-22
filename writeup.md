# **Traffic Sign Recognition** 

## Writeup

---

[//]: # (Image References)

[image1]: ./Image/visualization.png "Visualization"
[image2]: ./Image/sampleflatten.png "number of sample flatten"
[image3]: ./Image/preprocess1.png "default image"
[image4]: ./Image/preprocess2.png "grobal contrast normalization"
[image5]: ./Image/dataAugmentation1.png "Affine"
[image6]: ./Image/dataAugmentation2.png "contrast&brightness"
[image7]: ./Image/Sign.png "Traffic Sign 1"
[image8]: ./Image/bar1.png "Bar Chart Traffic Sign 1"
[image9]: ./Image/bar2.png "Bar Chart Traffic Sign 2"
[image10]: ./Image/bar3.png "Bar Chart Traffic Sign 3"
[image11]: ./Image/bar4.png "Bar Chart Traffic Sign 4"
[image12]: ./Image/bar5.png "Bar Chart Traffic Sign 5"

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ksks1986/P2_Traffic-Sign-Classifier/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
->Number of training examples is 34799.
* The size of the validation set is ?
->Number of validation examples is 4410.
* The size of test set is ?
->Number of test examples is 12630.
* The shape of a traffic sign image is ?
->Image data shape is (32, 32, 3).
* The number of unique classes/labels in the data set is ?
->Number of classes is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
These are histgrams showing the data distribution for each classes. 


![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I roughly flatten the number of examples per label by copying the minority images, because the number variation of each label is too large as see in histgram.
Here is a histgram after flattening.

![alt text][image2]

Next, I do the global contrast normalization which means subtraction and then standard deviation division for each image
because the images' contrast differ from each other.
Here is an example of a traffic sign image before and after normalization.

![alt text][image3]
![alt text][image4]

To add more data to the the data set, I used the following techniques in traing phase because padding out the data.
Firstly I use random affine transformation.
Here is an example of an original image and an augmented image:

![alt text][image3]
![alt text][image5]

Secondly I randomize the contrast and brightness.
Here is an example of an original image and an augmented image:

![alt text][image3]
![alt text][image6]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| (1)Input         		| 32x32x3 RGB image   							| 
| (2)Data Augmentation         		| random affine transform, contrast and brightness   							| 
| (3)Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| (4)Leaky ReLU					| alpha 0.01												|
| (5)Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| (6)Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| (7)Leaky ReLU		| alpha 0.01        									|
| (8)Max pooling		| 2x2 stride,  outputs 5x5x16        									|
| (9)dropout		| keep prob 0.9        									|
| (10)Fully connected		| input is flatten (1)(4)(7). inputs 4648, outputs 120        									|
| (11)Leaky ReLU		| alpha 0.01        									|
| (12)dropout		| keep prob 0.9        									|
| (13)Fully connected		| input 120, outputs 84        									|
| (14)Leaky ReLU		| alpha 0.01        									|
| (15)dropout		| keep prob 0.9        									|
| (16)Fully connected		| input 84, outputs 43(n_classes)        									|
| (17)Softmax				|         									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer, learning rate 0.0007, batch size 32 and number of epochs 70.
Parameter initialize is mean=0 and sigma=0.05.


Data Augmentation parameters are here:
*angle range = +/- 5 deg
*scale range = 0.5 to 1.5
*shift range = 5 pixels
*brightness max delta = 0.5
*contrast lower = 0.1
*contrast upper = 2.5


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
->99.6%
* validation set accuracy of ? 
->96.3%
* test set accuracy of ?
->94.4%

* What was the first architecture that was tried and why was it chosen?

->LeNet is chosen because LeNet is proven architecture for image recognition.

* What were some problems with the initial architecture?

->Validation accuracy dosen't reach 93% althogh training accuracy exceeds 93%.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

->Firstly I plot the learning curve and validation curve to see the model is overfitting or underfitting.
  When the model is overfitting, I increase data augmentation parameter and add the dropout to reduce overfitting.
  When the model is underfitting, I increase the model complexity by adding layers or nodes.
  The final architecture isn't much overfitting and underfitting.

* Which parameters were tuned? How were they adjusted and why?
  What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

->Dropout keep_prob was tuned 0.9 because training accuracy gets worse when I use less than 0.9.
Also, epoch and batch number, data augmentation parameters are tuned by trial and error for better validation accuracy.
Convolution layer works well for image recognition because less parameters and robustness of a few variation of image.
Dropout layer forces the model to reduce dependency of particular nodes, so it reduce overfitting.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web, trimmed and resized to 32*32 size:

![alt text][image7]

*The first image might be difficult to classify because this image is blur.
*The second image might be difficult to classify because the sign is almost coverd by snow.
*The third image might be difficult to classify because this image is blur.
*The forth image might be difficult to classify because this image is blur and tilted.
*The fifth image might be difficult to classify because this image is blur and lack a little.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit(100km/h)      		| Speed limit(50km/h)   									| 
| General caution     			| General caution 										|
| Speed limit(60km/h)					| Speed limit(60km/h)											|
| Go straight or right	      		| Go straight or right					 				|
| Right-of-way at the next inteersection			| Right-of-way at the next inteersection      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is approximately valid accuracy for the testset accuracy 94.4%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 58th cell of the Ipython notebook.

For the first image, the model has relatively the same predictions for "speed limit(50km/h, 80km/h, 100km/h)", and the image does contain a speed limit sign. 
Although the true limit number is 100km/h, the model predicted 50km/h because I think the blur of character makes miss-recognize.
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .33         			| Speed limit(50km/h)   									| 
| .26     				| Speed limit(80km/h) 										|
| .23					| Speed limit(100km/h)											|
| .07	      			| Speed limit(30km/h)					 				|
| .03				    | No passing for vehicles over 3.5 metric tons      							|


![alt text][image8]


For the second image, the model has strong confidence to predict "General caution"(probability of 0.93). This is the right prediction.
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .93         			| General caution   									| 
| .07     				| Traffic signals 										|
| .00					| Pedestrians											|
| .00	      			| Wild animals crossing					 				|
| .00				    | Road work      							|


![alt text][image9]


For the third image, the model has strong confidence to predict "Speed limit(60km/h)" (probability of almost 1.00). This is the right prediction.
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.00         			| Speed limit(60km/h)   									| 
| .00     				| Speed limit(80km/h) 										|
| .00					| Turn right ahead											|
| .00	      			| Speed limit(50km/h)					 				|
| .00				    | Go straight or right      							|

![alt text][image10]

For the forth image, the model is relatively sure that this is "Go straight or right"(probability of 0.42 against second prediction probability 0.18) than others.
But probability of 0.42 isn't strong confidence, the reliability of this prediction isn't much high.
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .42         			| Go straight or right   									| 
| .18     				| Priority road 										|
| .15					| Turn left ahead											|
| .11	      			| Ahead only					 				|
| .07				    | Keep right      							|

![alt text][image11]


For the fifth image, the model has strong confidence to predict "Right-of-way at the next intersetion"(probability of almost 1.00). This is the right prediction
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.00         			| Right-of-way at the next intersection   									| 
| .00     				| Beware of ice/snow 										|
| .00					| Road work											|
| .00	      			| Dangerous curve to the right					 				|
| .00				    | Bicycles crossing      							|

![alt text][image12]

