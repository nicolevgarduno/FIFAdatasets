# Analyzing FIFA player statistics
The purpose of this project was to conduct multiple manipulations to the data and ultimately build a machine learning model that could predict the overall value for each FIFA player based on the skillsets.
This project included uses of data cleaning, data engineering, data encoding, and construction of a pipeline. Along with these skills, the ML model was constructed in SparkML and Tensorflow.

## Part 1

This area consists of setting up a spark connection, uploading the dataset files, and generating a master schema from one of the datasets. Then, adding column into a spark dataframe. Resulting dataframe can be found in Section A.

## Part 2

Here, I focusing on analyzing the dataframe by finding the clubs with the most contracts, an average player's age, and most popular nationality.

## Part 3

This area is dedicated to building the machine learning model. In Section B, I am performing data cleaning. I drop unneccessary columns, as our main goal is to build a model that will predict the overal value for each FIFA player based on their skillsets. Then, I am cleaning columns by fixing timestamps, dropping unneccessary characters within columns, converting columns to the correct datatype, handling NULL values. In Section C, I am checking for outliers, converting my dataframe to Pandas to check correlation and dropping those which do have a high correlation. I then handle all values into their datatypes and perform encoding on datatypes which need it. In Section D, I construct my pipeline with a transformer, typecaster, indexer, further encoder, assembler, and scaler. I split my datasets into train and test data and transofrm the dataset to use it in the pipeline and prepare it for the ML model.

## Part 4
In Part 4, I create a linear regression model and train my train and test dataset with this model. The test accuracy is 97.44%. with a MSE of 4.52. Although these are great results, I chose to apply hyperparameter tuning through cross-validation. After cross-validation, my test MSE is 4.52. Therefore, I chose to keep my original linear regression model since there was not much of a difference between the before/ after using cross-validation. In Section E, I train a random forest regressor model and achieve a test accuracy of 99% with a MSE of 0.66.

## Part 5

In this area, I began building a ML model using TensorFlow. In Section F, I create a MLP with a test accuracy of about 97% with a MAE of 1.50. In Section G, I create a CNN with a test accuracy of 97.86% and a MAE of 0.909.
## Outcome

My best model was my random forest regressor model with numTrees = 15 and maxDepth = 10, which gave me a test dataset accuracy of about 99% and a test MSE of 0.6637, which was better than the performance of the linear regression model (~97%, 4.51 test MSE), the Multi-Layer Perceptron model (~97% test accuracy, 1.50 test MAE), and the Convolutional Neural Network Model (~97%, 0.909 test MAE). This is likely because of the ability of random forest regressor to closely fit to the patterns in the data. I originally believed that the random forest model would be overfit to the training data, however it seems that it was still able to model the test dataset quite well. This is likely due to the choice of features to use in the training, as I chose a select number of features that I identified as having low correlations with each other and also that seemed most relatable to identifying the overall value of individual players. By using only a small set of specific features, we likely prevented the model from becoming too complex by omitting columns and variables that were too position or player specific, allowing us to design a model that could generalize well to unseen datasets.

## Tech Stack

Python, SparkML, Tensorflow
