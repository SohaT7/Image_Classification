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
#### Model Selection:
The file [model_comparison](#https://github.com/SohaT7/Image_Classification/blob/main/model_comparison.ipynb) contains code that builds multiple models and compares the performance of the metrics for each. The models and the combination of letters they are referred to by in this project (these do not necessarily coincide with the actual official names/acronyms for each model in the field) are as follows:
 - Linear model (lm)
 - Neural Network model (nn)
 - Deep Neural Network model (dnn)
 - Deep Neural Network with a dropout layer (dndd)
 - Convolutional Neural Network (cnn)
 - Deep Convolutional Neural Network (dcnn)

After building and training each model, the 'accuracy' metric for the train, validation, and test sets was plotted for each model, as shown below:

<img width="400" alt="acc_nn" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/nn_graph.png">

<div class="row">
    <div class="column">
        <img width="200" alt="acc_dnn_dnnd" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/dnn_dnnd_graph.png"> 
    </div>
    <div class="column">
        <img width="200" alt="acc_cnn_dcnn" src="https://github.com/SohaT7/Image_Classification/blob/main/Resources/Images/cnn_dcnn_graph.png"> 
    </div>
</div>

The performance of each model was determined by calculating the 'loss' and 'accuracy' metrics, based on which the top 3 best performing models were chosen: the Deep Neural Network with a Dropout layer (dnnd), the Convolutional Neural Network (cnn), and the Deep Convolutional Neural Network (dcnn).

#### Image Classification:

## Results


## Summary


## Contact Information
Email: st.sohatariq@gmail.com

<img width="700" alt="image" src="https://github.com/SohaT7/Mission_to_Mars/blob/main/Images/page1.png"> 
