# Image_Classification
## Table of Contents
- [Overview of the Analysis](#overview-of-the-analysis)
    - [Purpose](#purpose)
    - [About the Dataset](#about-the-dataset)
    - [Tools Used](#tools-used)
    - [Description](#description)
        - [Model Comparison](#Model-Comparison)
        - [Building Models](#Building-Models)
        - [Compiling Models](#Compiling-Models)
        - [Model Architectures](#Model-Architectures)
        - [Model Metrics](#Model-Metrics)
- [Results](#results) 
    - [Deep Neural Network (DNN)](#Deep-Neural-Network)
    - [Convolutional Neural Network (CNN)](#Convolutional-Neural-Network)
    - [Deep Convolutional Neural Network (DCNN)](#Deep-Convolutional-Neural-Network)
    - [Making an Example Prediction](#Making-an-Example-Prediction)
- [Summary](#summary)
- [Contact Information](#contact-information)


## Overview of the Analysis
### Purpose:
This Computer Vision analysis aims to solve an image classification problem through building and evaluating machine learning models - Deep Neural Network (DNN), Convolutional Neural Network (CNN), and Deep Convolutional Neural Network (DCNN) models - using 60,000 labeled images of handwritten digits from 0-9 in the MNIST dataset.

### About the Dataset:
The MNIST (Modified National Institute of Standards and Technology) dataset is a popular dataset containing 60,000 images of handwritten digits from 0-9. The images are grayscale and 28 pixels by 28 pixels in dimension, which totals up to 784 pixels. Each pixel value is an integer between 0 and 255, with higher numbers signifying a darker intensity. The training set has 785 columns: the first column is for the label of the image, and the remaining 784 are each for the 784 pixel values. The test set has 784 columns for the pixel values (i.e. there is no column for the label of the image). The training set comprises of 50,000 images whereas the validation and test sets comprise of 10,000 images. 

### Tools Used:
* Google Cloud Platform (GCP)
* APIs - Vertex AI API, Notebooks API
* Python (TensorFlow, Keras, NumPy, Matplotlib, Seaborn libraries)

### Description:
Google Cloud Platform was used for data modeling. The APIs used include the Vertex AI API and the Notebooks API primarily. While creating the [image_classification](https://github.com/SohaT7/Image_Classification/blob/main/image_classification.ipynb) notebook - the main file for this project - in the Google Cloud Platform, the 'Without GPUs' option was selected, as shown below.

![Notebook](https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/notebook.png)

#### Model Comparison: 
Multiple models were built and the performance of their metrics compared (see the [model_comparison](https://github.com/SohaT7/Image_Classification/blob/main/model_comparison.ipynb) file for reference). The models, and the combination of letters with which they are referred to in this project for ease of reference (these do not necessarily coincide with the actual official names/acronyms for each model in the field), are as follows:
 - Neural Network model (nn)
 - Deep Neural Network model (dnn)
 - Deep Neural Network with a dropout layer (dndd)
 - Convolutional Neural Network (cnn)
 - Deep Convolutional Neural Network (dcnn)

After building and training each model, the 'accuracy' metric for the training, validation, and test sets was plotted for each model, as shown below:

<img style="width:60%" alt="acc_nn" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/nn_graph.png">

<div class="row">
    <div class="column">
        <img style="width:60%" alt="acc_dnn_dnnd" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/dnn_dnnd_graph.png"> 
    </div>
    <div class="column">
        <img style="width:60%" alt="acc_cnn_dcnn" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/cnn_dcnn_graph.png"> 
    </div>
</div>

The performance of each model was determined by calculating the 'loss' and 'accuracy' metrics, based on which the top 3 best performing models were chosen: the Deep Neural Network with a Dropout layer (dnnd), the Convolutional Neural Network (cnn), and the Deep Convolutional Neural Network (dcnn).

#### Building Models:
This project builds and evaluates the performance of a Deep Neural Network, Convolutional Neural Network, and a Deep Convolutional Neural Network model. The code for it can be found in the [image_classification](https://github.com/SohaT7/Image_Classification/blob/main/image_classification.ipynb) notebook. 

A model is built by adding layers to it. Each model herein consists of a flattened layer, dense layer(s), and a dense (output) layer. A flattened layer converts a multi-dimensional input tensor into a 1-dimensional array as input for the next layer. A dense layer is defined as a neural network layer where each neuron has a weighted connection to each neuron in the next layer. A dense (output) layer consists of labeled classes, which in our case are 10 - one for each digit from 0-9. 

The models are built using a Sequential model. A Sequential model is a linear model which consists of layers connected to one another, such that output from one layer is input to the next one. If a model seems to be overfitting the data, the regularisation technique is commonly used. Adding a dropout layer to a model is an example of regularisation. At each training interation, the dropout layer drops random neurons from the network with a probability p - we have defined p as 0.25 (at 25%) in our models here.

A Convolutional Neural Network (CNN) additionally consists of convolutional and pooling layers. A convolutional layer helps recognize patterns and extract features; the model learns the parameters in this layer. A pooling layer reduces the dimensionality of inputs coming in from the previous hidden layer. In a CNN, the convolutional and pooling layers taken together recognize patterns and reduce dimensionality of the image before placing it as a flattened input into a DNN. This helps reduce the training times and improve performace (as can be seen by comparing our DNN and CNN models here too). 

Activation function ReLu (Rectified Linear Unit) is used in these models for it does not activate all neurons at the same time and thus prevents the exponential growth in computation required otherwise to operate the network model. A Softmax function is used as the activation function for the dense (output) layer herein for it returns probabilities instead of logits, i.e. it returns a value between 0 and 1 against each class instead of a value between positive infinity and negative infinity. "Epochs" are the number of complete passes through the training dataset. "Batch size" is the number of samples processed before the model is updated. We use 20 epochs and 32 as the batch size for our models here.

#### Compiling Models:
After a model has been built by adding layers to it, it is then compiled by specifying the optimizer, loss function, and metrics for the model. An Optimizer determines how the model is updated based on the data provided and its loss function. Adam Optimizer, one of the most popular optimizers, is used in these models. A loss function measures how accurate the model is during training. Since we are dealing with a classification problem and have 2 or more labeled classes (we have 10 classes), we will be using the "Sparse Categorical Crossentropy" as the loss function. Metrics monitor the datasets. The 'accuracy' metric calculates how often the predicted label equals the actual label. 

#### Model Architectures:
The model architectures for the Deep Neural Network with a Dropout layer (dnnd), Convolutional Neural Network (cnn), and Deep Convolutional Neural Network (dcnn) models are shown below, respectively:

<img style="width:60%" alt="dndd_summ" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/dnnd_summ.png"> 

<img style="width:60%" alt="cnn_summ" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/cnn_summ.png">

<img style="width:60%" alt="dcnn_summ" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/dcnn_summ.png">

#### Model Metrics:
After the model has been built and compiled, it is then "fit" or "trained" using the training dataset. The validation set is used to help find the optimal values for the hyper-parameters of the model (hyper-parameter optimisation) and thereby help with model selection. The 'loss' and 'accuracy' metrics from the training set and the 'val_loss' and 'val_accuracy' metrics from the validation set plotted against the number of epochs for the three models can be seen below:

<img style="width:70%" alt="dnnd_metrics" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/dnnd_metrics.png">

<img style="width:70%" alt="cnn_metrics" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/cnn_metrics.png">

<img style="width:70%" alt="dcnn_metrics" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/dcnn_metrics.png">

If loss does not go down smoothly, it signifies that either the batch size or the optimizer settings can be improved. Here, loss seems to go down fairly smoothly more or less. If the validation loss goes down but then starts to increase, it signifies that overfitting is starting to happen, i.e. the network has started to memorize details of the training set (the "noise") which do not occur in the validation set. In order to rectify this, we can either decrease the number of epochs or add regularization. Adding a dropout layer is a regularisation technique, which we have added to each of our models here. 

The 'accuracy' of the DNN model can be seen below, followed by that of the CNN and DCNN models:

<img style="width:60%" alt="cnn_dcnn_graph" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/dnnd_graph.png">

<img style="width:60%" alt="cnn_dcnn_graph" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/cnn_dcnn_graph.png">

## Results
The precision score, recall score, and confusion matrices for the three models are given below. The precision score tells us what proportion of positive identifications were actually correct, whereas the recall score tells us what proportion of actual positives was identified correctly. The confusion matrix plots the probabilities for all possible events occurring between predicted and actual values for all labels. 

### Deep Neural Network:
* Precision Score: 0.9804 - 98.0% of predicted true are actually true.
* Recall Score: 0.9803 - 98.0% of actually true were predicted as true.

<img alt="CM_dnnd" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/CM_dnnd.png">

### Convolutional Neural Network:
* Precision Score: 0.9909 - 99.1% of predicted true are actually true.
* Recall Score: 0.9909 - 99.1% of actually true were predicted as true.

<img alt="CM_cnn" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/CM_cnn.png">

### Deep Convolutional Neural Network:
* Precision Score: 0.9934 - 99.3% of predicted true are actually true.
* Recall Score: 0.9934 - 99.3% of actually true were predicted as true.

<img alt="CM_dcnn" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/CM_dcnn.png">

### Making an Example Prediction:
An example prediction made on the first image in the dataset, using the Deep Convolutional Neural Network (DCNN), accurately predicts the digit.

<img style="width:50%" alt="prediction" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/prediction.png">

## Summary
Out of the three models - i.e. Deep Neural Network (DNN), Convolutional Neural Network (CNN), and Deep Convolutional Neural Network (DCNN) - the DCNN model performs the best, with the highest precision and recall scores. Its precision score is 99.3%, i.e. out of all the values predicted to be 'True', 99.3% are actually true too. Its recall score is 99.3% as well, i.e. out of all the actually true values, the model is able to predict 99.3% of them as 'True'. 

## Contact Information
Email: st.sohatariq@gmail.com

 
