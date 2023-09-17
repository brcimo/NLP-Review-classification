# NLP-Review-classification

The goal for this data analysis is to see if it is possible to predict if someone has written a positive or negative review of a movie based on their statements by using a sentiment analysis model.

One type of neural network capable of performing text classification is a Deep Learning Neural Network (DNN). “Deep Neural Networks (DNNs) are considered Feed Forward Networks in which data flows from the input layer to the output layer without going backward, and the links between the layers are one-way. This means this process goes forward without touching the node again” (Mohapatra, 2022). A DNN typically contains many hidden networks in between the input and output layers. Each layer applies weights and biases to the data that are learned in the training process. For text classification, the text data needs to be converted to numeric data to be processed by the neural network. This numeric data is fed into the model with an embedding layer before being fed into the following layers.

The following is the exploratory data analysis performed:

1.	The text data was stripped of all non-English characters using regular expressions. The expression used removed everything except letters, numbers, and whitespace. All the text was also converted to lowercase. The following is the code used to do this with annotations:

2.	The vocabulary size was computed using the Tensorflow Tokenizer API. All the words in the data set are converted to tokens by individual word. To get the vocabulary size, we simply find the length of the list of words. The following is the code used to do this with annotations:

3.	For this analysis, the keras Embedding layer was used. “…in an embedding, words are represented by dense vectors where a vector represents the projection of the word into a continuous vector space. The position of a word within the vector space is learned from text and is based on the words that surround the word when it is used. The position of a word in the learned vector space is referred to as its embedding” (Brownlee, 2017). The layer uses the integer data set that was created using the Tokenizer class. The padded token sequences were generated using pad_squences. The following is the code used to do this with annotations:

4.	The maximum sequence length was chosen by finding the longest review by number of words. After the non-English words and characters were removed the longest sequence of words was found and the rest of the sentences are padded to provide a uniform input size for the model. The maximum length found for a review does not cause any significant drain on system memory or training efficiency.

The following are the steps used to prepare the data:

1.	Load the data set from the text file into a pandas Dataframe.
2.	Check the data types and get descriptive statistics.
3.	Check for Null or NA values and fix if necessary.
4.	Check for duplicate values and fix as necessary.
5.	Clean the data by removing non-english characters and symbols.
6.	Get the number of words in the data set and determine the length for padding.
7.	Tokenize the review data.
8.	Split the review and label data into train and test sets with an 80/20 split.
9.	Convert the text sequences into a sequence of integers and add padding to make each sequence the same size.
10.	Convert the label lists into numpy arrays.

The model developed appears to perform at an acceptable level, but it is not optimal. The model is not overfitting, you can see in the loss graph that loss is decreasing to near zero and the value loss stays flat. An optimal model would have both values decreasing to near zero. To prevent overfitting, the early return call back was used along with dropout between hidden layers. Early stopping helps to stop the model from overfitting during training. It works by monitoring a value and stops if that value does not continue to improve. Dropout is a regularization technique that sets a percentage of input units to zero to help prevent overfitting. The model accuracy was assessed using an 80/20 training/testing data set. The model accuracy after training has a training accuracy of 99.75% and test accuracy of 77.5%. I believe the model could be improved by adding more data to the training set.

Brownlee, J. (2017, October 3). How to Use Word Embedding Layers for Deep Learning with Keras. Machine Learning Mastery. https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

Mohapatra, S. (2022, November 18). Analyzing and Comparing Deep Learning Models. Analytics Vidhya. https://www.analyticsvidhya.com/blog/2022/11/analyzing-and-comparing-deep-learning-models/


Copyright 2023 by Bryan Cimo see LICENSE for terms.