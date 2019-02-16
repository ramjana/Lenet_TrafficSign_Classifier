# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report



[//]: # (Image References)

[image1]: ./output_images/sample_data_image.png "Image Visualization"
[image2]: ./output_images/dataset_occurences.png "Train Count Visualization"
[image3]: ./output_images/augmented_images.png "Augmented image Visualization"
[image4]: ./output_images/accuracy_curve.png "Validation accuracy curve"
[image5]: ./output_images/y_crossing.jpg "New signs visulaization"
[image6]: ./output_images/30_speedlimit.jpg "New signs visulaization"
[image7]: ./output_images/Yield.jpg "New signs visulaization"
[image8]: ./output_images/nparking.jpg "New signs visulaization"
[image9]: ./output_images/workzone.jpg "New signs visulaization"
[image10]: ./output_images/softmax_output.png "softmax outputs of new signs"



---


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

To have higer degress of success in training and validating model, we need to have lots of datasets This is however is not possible In such cases, data augmentation helps us generate additional training examples. We will generate additional data samples by applying affine transformation to the image. Affine transformations refer to transformations that do not alter the parallelism of lines, i.e. can be represented as a linear operation on the matrix. We will specifically use rotation, shearing and translation to simulate the effect of viewing the sign from different angles and different distances. Figure below presents original image and augmented images generated from it.


Good article on data augmentation. I am using some the techniques written in article

https://github.com/vxy10/ImageAugmentation

 

Below are sample augmented images randomly chosen from input.

![alt text][image3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:


input layer: [.,32X32X3]
layer: Conv0 -> RELU -> [,32,32,3]
layer: Conv1 -> RELU : [.,16,16,32]
layer: Conv2 -> RELU -> MaxPool: [.16,16,64]
layer: Conv2 -> RELU: [.,16,16,64]
layer: Conv3 -> RELU -> MaxPool: [.,8,8,64]
layer: Conv4 -> RELU: [.,8.,8.,128] 
layer: Conv5 -> RELU -> MaxPool: [.,4.,4.,128]
layer: FC1 -> ReLu: [.,1024]
layer: FC2 -> Relu: [.,1024]
output layer: FC -> ReLu: [.,43]


 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer with learning rate o 1e-4. I have split training data into train and validation sets. I have run for 20 epocs and in each epoch i ran model for 10000 iterations or till i dont see improvement in accurancy.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.6
* validation set accuracy of 99.7
* test set accuracy of 97.3


 I have started with  LeNet provided by Udacity. This model was proved to work well and provided close to 80% accuracy for traffic sign classification. I needed something better. I started tweaking the model by adding more hidden layer(s) to first three convolution layer.
I started seeing accuracy go higher. To improve more, , I added two dropout layers with 0.5 keep probability and increased the training epochs to 22. The final model is described as follows:


input layer: [.,32X32X3]
layer: Conv0 -> RELU -> [,32,32,3]
layer: Conv1 -> RELU : [.,16,16,32]
layer: Conv2 -> RELU -> MaxPool: [.16,16,64]
layer: Conv2 -> RELU: [.,16,16,64]
layer: Conv3 -> RELU -> MaxPool: [.,8,8,64]
layer: Conv4 -> RELU: [.,8.,8.,128] 
layer: Conv5 -> RELU -> MaxPool: [.,4.,4.,128]
layer: FC1 -> ReLu: [.,1024]
layer: FC2 -> Relu: [.,1024]
output layer: FC -> ReLu: [.,43]


 Training:

Model HyperParameters : 
 - Learning rate =1e-3, keep_prob= 0.5 (dropout layer), L2 regularization weight (10^(-5)) BATCH_SIZE=512
 - Training / Validation data : Validation data is generated from train_test_split() with 25% of train data is used as validation data

I trained model for 20 epochs and in each epoch running model for 5000-10000 iterations; I break off training if I dont see improvement in accuracy for each epochs, I have augmented 5-10 images per train_image. After 5 epocs, I have reduced augmentation factors by 0.9

outputs of training iteration:


Training Loop # 0

#200, Train Acc.:  43.0%, Val Acc.:  50.3%, Test Acc.:  45.1%
#400, Train Acc.:  62.3%, Val Acc.:  80.0%, Test Acc.:  69.1%
#600, Train Acc.:  85.4%, Val Acc.:  93.9%, Test Acc.:  85.3%
#800, Train Acc.:  90.4%, Val Acc.:  97.5%, Test Acc.:  90.0%
#1000, Train Acc.:  93.2%, Val Acc.:  98.4%, Test Acc.:  92.4%
#1200, Train Acc.:  96.7%, Val Acc.:  99.1%, Test Acc.:  92.9%
#1400, Train Acc.:  97.3%, Val Acc.:  99.1%, Test Acc.:  93.7%
#1600, Train Acc.:  98.4%, Val Acc.:  99.3%, Test Acc.:  94.3%
#1800, Train Acc.:  97.9%, Val Acc.:  99.5%, Test Acc.:  94.8%
#2000, Train Acc.:  98.4%, Val Acc.:  99.5%, Test Acc.:  95.3%
#2200, Train Acc.:  98.8%, Val Acc.:  99.6%, Test Acc.:  95.3%
#2400, Train Acc.:  98.8%, Val Acc.:  99.4%, Test Acc.:  95.2%
#2600, Train Acc.:  98.4%, Val Acc.:  99.6%, Test Acc.:  95.7%
#2800, Train Acc.:  98.8%, Val Acc.:  99.5%, Test Acc.:  95.9%
#3000, Train Acc.:  99.0%, Val Acc.:  99.4%, Test Acc.:  95.7%
#3200, Train Acc.:  99.2%, Val Acc.:  99.6%, Test Acc.:  96.3%
#3400, Train Acc.:  99.4%, Val Acc.:  99.5%, Test Acc.:  95.8%
#3600, Train Acc.:  99.2%, Val Acc.:  99.7%, Test Acc.:  95.7%
#3800, Train Acc.:  99.2%, Val Acc.:  99.5%, Test Acc.:  96.5%
#4000, Train Acc.:  99.0%, Val Acc.:  99.6%, Test Acc.:  96.0%
#4200, Train Acc.:  99.0%, Val Acc.:  99.7%, Test Acc.:  96.0%
#4400, Train Acc.:  98.8%, Val Acc.:  99.7%, Test Acc.:  96.2%
#4600, Train Acc.:  99.6%, Val Acc.:  99.6%, Test Acc.:  96.2%
#4800, Train Acc.:  98.6%, Val Acc.:  99.5%, Test Acc.:  96.0%
#5000, Train Acc.:  99.4%, Val Acc.:  99.6%, Test Acc.:  96.3%
#10000, Train Acc.:  99.6%, Val Acc.:  99.7%, Test Acc.:  97.3%


Here are accuracy curve from training and testing model.
![alt text][image4]



Accuracy on test set:  97.3%


##### How did you choose the model? Was it a previous model known to work well with this problem or you have you created a new one, if you have created a new one, why?

   I have started with provided udacity 5 layer model (conv_layer0->conv_layer1->conv_layer2->fc->fc)  and added another fc layer and that got me 90% test accuracy.  I have used cross-validation approach (splitting train images into validation& training images) added data augmentation stage to the pipeline. It helped push the accuracy to 92% but test accuracy still hovered around 90%. 
   
   I researched web and found going deeper along with batch-normalization , drop out layer helps to improve accuracy. I have experimented adding more convolution layers and finally settle on the model that helped me to improve accuracy to 97%+.

##### How to choose the optimizer? What are the Pros & Cons?

I chose most widely used first order optmization technique Gradient Descent that converges pretty fast on large data sets. There are many first order optimiation techniques.

  SGD is most basic and low learning rate will make the learning slow while a large learning rate might lead to oscillations. SGD takes long time oscillates when curves are steeply in one dimension than in another. Using momentum helps to accelrate the learning process.
  
  Good article on optmizer
  
  http://ruder.io/optimizing-gradient-descent/
  

##### How have you tuned the hyperparameter? How many values have you tested?

I have started with simple approach of chossing learning rate of 1e-3 and epochs 50 and trained model using cross validation sets. I have started adjusting learning rate (scaling down mostly) after 5 epochs and scaled-down factor 10-20% after every 5 epochs.

Tried with  drop probablity range of values 0.3-0.5. I choose the current version of training i.e iterations of 5000-10000 for each epoch till i dont see improvement. 

Added training loss and accuracy arrays to register loss,accuracy during training and validation data sets. 

##### How have you measured your model performance?

Using cross validation and test images, I have tested the accuracy of the prediction. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are  German traffic signs that I found on the web:


I chosse these test images for various reasons. Mainly to see how my model would perform on images that have different backgrounds, more than one image, angled and different contrast / brightness.

workzone - sign has highest brightness (more light exposed) with background tree. 


noParking - Sign has texts and with dark backgroud
![alt_text][image8]
Yield  - Sign is angled towards right side
![alt_text][image7]
30-speedlimit - Sign background images, partial image of sign.
![alt_text][image6]
y-crossing - Sign with low contrast and background images
![alt text][image5]
workzone -   Sign has background images.
 ![alt_text][image9]
 
The model perform poorly with Germany sign-data. It did ok with signs that resemble with Germany signs.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Y-crossing      		| Road narrows on the right 					| 
| No-Parking     		| Yield             							|
| Yield					| Yield											|
| WorkZone	      		| Road work  					 				|
| Speed-limit			| Road Work         							|



 top five softmax probabilities of the predictions on the captured images are outputted
 

![alt_text][image10]



