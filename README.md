# image-orientation-classifier
We have implemented a fully-connected feed-forward network to classify image orientation with the backpropagation algorithm to train the network using gradient descent. Provided with a dataset of images from the Flickr photo sharing website.
Contributers :  1. Pranav Pande (pmpande)
                2. Nishant Shah (nishshah)
 
We have implemented KNN and ANN in seperate files and calling their appropriate methods in orient.py file.
 
Repository also includes trained weights and incorrect outputs from ANN using best configuration.

# KNN
In KNN we are taking Euclidian distance.
But for sake of less complexity in running time, we are not finding the square root for finding distance, and keeping square value of it.

Training KNN takes a lot of time in local machine(Around 80-90 mins) as we haven't done any optimization.

# ANN
We started ANN with constant weight 1 to all weights but this resulted in convergence of fitness too early and also for every output node.
So we explored resources to know what should be ideal initial weight distribution and what number of hidden layers should be used.
We then came up with a solution of initial weight to be random number between (-1/sqrt(i),1/sqrt(i)) where i is the number of input nodes and number of hidden nodes to be mean of input and output nodes. But this solution didn't work out well and requires long time for training.
So we went for trials and errors for initial weight distribution and number of hidden nodes.
All results of these experiments are well documented in PDF report.
Every training of ANN also produces a file 'trained_weights' which eases multiple running of test for same configured neural network.
'trained_weights' file contains weights of every edge.
Every test run produces a file "incorrect" which contains images classified incorrectly.

About calculation:
We are not using matrix multiplication or any external library to speed up our code. Our single training completes running in 2-3 mins.

# Best
For best case, we are using ANN. Configuration and result of it is explained in PDF report.
We have cached the weights of this ANN in 'model_file'. So running the best program will test the test data instantly.
