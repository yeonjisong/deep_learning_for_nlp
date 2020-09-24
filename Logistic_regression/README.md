# DLNLP-LogisticRegression
The goal is to perform twitter sentiment analysis for sentence-level text based on logistic regression with gradient descent. First, clean the data and perform feature extraction. Then learn basic training and testing concepts and implement a sentiment classifier.

## Getting started
### Data Preprocessing and Analysis
Modify and execute the preprocess.py file. First, read the .csv fie and then build the vocabulary dictionary with data statistics.

### Feature Extraction & Normalization
Change each sentence into features using bag-of-word (BoW), which is a sentence representation method based on word counting but disregards grammar and even word order.

### Logistic Regression
Derive the cross entropy loss, gradient descent and backpropagation.

### Implementation
Implement Sigmoid function and cost function on the logistic regression.py file to train a sentiment classifier.

## Running the tests
Load the data from [preprocess.py](preprocess.py).
run the data using logistic regression [logistic_regression.py](logistic_regression.py).

## Built With
* [PyTorch](-) - 
* [Numpy](-) - 
* [Counter](https://pymotw.com/2/collections/counter.html) - 
