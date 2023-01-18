# Image_Classification
## Table of Contents
- [Overview of the Analysis](#overview-of-the-analysis)
    - [Purpose](#purpose)
    - [About the Dataset](#about-the-dataset)
    - [Tools Used](#tools-used)
    - [Description](#description)
- [Results](#results)
- [Summary](#summary)
- [Contact Information](#contact-information)


## Overview of the Analysis
### Purpose:

### About the Dataset:

### Tools Used:

### Description:
#### Model Comparison:
The file [model_comparison](#https://github.com/SohaT7/Image_Classification/blob/main/model_comparison.ipynb) contains code that builds multiple models and compares the performance of the metrics for each. The models and the combination of letters they are referred to by in this project (these do not necessarily coincide with the actual official names/acronyms for each model in the field) are as follows:
 - Neural Network model (nn)
 - Deep Neural Network model (dnn)
 - Deep Neural Network with a dropout layer (dndd)
 - Convolutional Neural Network (cnn)
 - Deep Convolutional Neural Network (dcnn)

After building and training each model, the 'accuracy' metric for the train, validation, and test sets was plotted for each model, as shown below:

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
The Deep Neural Network with a Dropout layer (dnnd) consists of a Flatten layer, 1 dense hidden neuron layer, and 1 dense output layer. The Flatten layer ....
The output layer has 10 classes, 1 for each digit from 0-9.

<img style="width:60%" alt="dndd_summ" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/dnnd_summ.png"> 

The Convolutional Neural Network (cnn) consists of a ...

<img style="width:60%" alt="cnn_summ" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/cnn_summ.png">

The Deep Convolutional Neural Network (dcnn) is made up of ...

<img style="width:60%" alt="dcnn_summ" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/dcnn_summ.png">

#### Model Metrics:
The 'loss' and 'accuracy' metrics for the train set and the 'val_loss' and 'val_accuracy' metrics for the validation set plotted against the number of epochs for the three models can be seen below:
<img style="width:70%" alt="dnnd_metrics" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/dnnd_metrics.png">

<img style="width:70%" alt="cnn_metrics" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/cnn_metrics.png">

<img style="width:70%" alt="dcnn_metrics" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/dcnn_metrics.png">

The 'accuracy' of the CNN and DCNN can be compared below:
<img style="width:60%" alt="cnn_dcnn_graph" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/cnn_dcnn_graph.png">

## Results
The precision score, recall score, and confusion matrices for the three models are given below.
### The Deep Neural Network with a dropout layer (dnnd):
Precision Score: 0.9790
Recall Score: 0.9789

<img alt="CM_dnnd" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/CM_dnnd.png">

### The Convolutional Neural Network (cnn):
Precision Score: 0.9928
Recall Score: 0.9928

<img alt="CM_cnn" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/CM_cnn.png">

### The Deep Convolutional Neural Network (dccn):
Precision Score: 0.9908
Recall Score: 0.9908

<img alt="CM_dcnn" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/CM_dcnn.png">

## Summary
An exmaple prediction was made on the first image in the dataset, which accurately predicts the digit.

<img style="width:50%" alt="prediction" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/prediction.png">


## Contact Information
Email: st.sohatariq@gmail.com

 
