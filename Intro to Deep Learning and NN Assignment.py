# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:58:29 2024

@author: aliso
"""

import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 

#from keras import metrics

concrete_df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_df.head()
# Extract Features and Target from the Data frame
data_columns = concrete_df.columns
features = concrete_df[data_columns[data_columns != "Strength"]]
target = concrete_df["Strength"]
n_cols = features.shape[1]

# Build a baseline model

def regression_model():
    """ A Function for the Regression Model.
    This is a Neural Network regression function with
    one hidden layer of 10 nodes,
    adam optimizer, and
    mean_squared_error loss function"""
    
    r_model = Sequential()
    r_model.add(Dense(10, activation ='relu', input_shape=(n_cols,)))
    r_model.add(Dense(1))
    # for training, using 'adam' optimizer
    r_model.compile(optimizer='adam', loss = 'mean_squared_error') 

    return r_model

############## Part A #################################

# Split and Train, Test Data 50 times and create a list of 50 mean squared errors.

times = 50
mse_list =[]  # for list of 50 Mean Squared Errors
    
for i in range(0, times):
        
        # split data
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.30, random_state=10)
        
        # call the regression function to create model
        r_model = regression_model()
        
        # Fit the regression model: train the model
        r_model.fit(x_train, y_train,  epochs = 50, verbose=2)

        # make predictions using the trained model with the test set
        predictions = r_model.predict(x_test)
        
        # transform 'predictions' and 'y_test' to lists so can calculate mse
        prediction_list = predictions.tolist()
        y_test_list = y_test.tolist()

        # calculate the mean squared error
        mse = mean_squared_error(y_test_list,prediction_list)
        # append to a list
        mse_list.append(mse)
        
# Checking the length of the mse_list - should be 50

print("Length of the mse_list: ", len(mse_list))

# Function for Mean of MSE list and for Standard Deviation

def mean_s(n):
    """

     Mean function for the Mean Squared Erros
     Parameter n: a list of numbers that represents the mse
     
    """    
    
    total = 0   # the total sum
    n_len = len(n)  # gives number of entries
    
    for i in range(0,n_len):
        total = total + n[i] 

    num_mean = total/n_len
    
    return num_mean

def stdev_s(n):
    """
    Standard Deviation function for the mean squared errors
    Parameter n: a list of numbers that represents the mse
    standard deviation = sqrt((xi - x_mean)^2/N)
    """   
    # get length of list:
    n_len = len(n)
        
    # Get the Mean of the list by dividing the total by the length of the list:
    num_mean = sum(n)/n_len   
    
    # Get squared sum of list: (xi - x_mean)^2
    sqr_sum = 0
    for i in range(0, n_len):
        sqr_sum = sqr_sum + (n[i] - num_mean)**2      
        
    # Divide by length and take square root:
    N = n_len-1
    s_std = (sqr_sum/N)**(1/2)
    
    return s_std

# Report the mean and the standard deviation of the mean squared errors.
mse_mean = mean_s(mse_list)
standard_dev_mse = stdev_s(mse_list)

print("The Mean of the Mean Squared Error for Un-normalized data : ", mse_mean)
print("The Standard Deviation of the Mean Squared Errorfor Un-normalized data: ", standard_dev_mse)


########### Part B ##########################

# Normalize the Data:
normalized_features = (features - features.mean())/features.std()
normalized_features.head()

# Save the number of features into n_cols for the network building
n_cols = normalized_features.shape[1]



# With Normalized Data:
# Repeat split, train, test 50 times, calculate MSE

times = 50
mse_list =[]  # for list of 50 Mean Squared Errors
    
for i in range(0, times):
        
        # split data
        x_train, x_test, y_train, y_test = train_test_split(normalized_features, target, test_size=0.30, random_state=10)
        
        # call the regression function to create model
        r_model = regression_model()
        
        # Fit the regression model: train the model
        r_model.fit(x_train, y_train,  epochs = 50, verbose=2)

        # make predictions using the trained model with the test set
        predictions = r_model.predict(x_test)
        
        # transform 'predictions' and 'y_test' to lists so can calculate mse
        prediction_list = predictions.tolist()
        y_test_list = y_test.tolist()

        # calculate the mean squared error
        mse = mean_squared_error(y_test_list,prediction_list)
        # append to a list
        mse_list.append(mse)
        
# Checking the length of the mse_list - should be 50

print("Length of the mse_list: ", len(mse_list))


# Report the mean and the standard deviation of the mean squared errors.
mse_mean_norm = mean_s(mse_list)
standard_dev_mse_norm = stdev_s(mse_list)

print("The Mean of the Mean Squared Error for Normalized data : ", mse_mean_norm)
print("The Standard Deviation of the Mean Squared Errorfor Normalized data: ", standard_dev_mse_norm)
print(" The Un-normalized data has an almost twice as large standard deviation.")
print(" The Mean is almost the same for the Normalized and Un-Normalized data.")


####################### Part C ##############################


# Normalized Data from Part B With 100 Epochs:
# Repeat split, train, test 50 times, calculate MSE

times = 50
mse_list =[]  # for list of 50 Mean Squared Errors
    
for i in range(0, times):
        
        # split data
        x_train, x_test, y_train, y_test = train_test_split(normalized_features, target, test_size=0.30, random_state=10)
        
        # call the regression function to create model
        r_model = regression_model()
        
        # Fit the regression model: train the model - 100 epochs
        r_model.fit(x_train, y_train,  epochs = 100, verbose=2)

        # make predictions using the trained model with the test set
        predictions = r_model.predict(x_test)
        
        # transform 'predictions' and 'y_test' to lists so can calculate mse
        prediction_list = predictions.tolist()
        y_test_list = y_test.tolist()

        # calculate the mean squared error
        mse = mean_squared_error(y_test_list,prediction_list)
        # append to a list
        mse_list.append(mse)


        
# Checking the length of the mse_list - should be 50
print("Length of the mse_list: ", len(mse_list))


# Report the mean and the standard deviation of the mean squared errors.
mse_mean_hundred = mean_s(mse_list)
standard_dev_mse_hundred = stdev_s(mse_list)

print("The Mean of the Mean Squared Error with 100 Epochs : ", mse_mean_hundred)
print("The Standard Deviation of the Mean Squared Errorfor Normalized data: ", standard_dev_mse_hundred)

### Comparing Mean and Stdev of 50 epochs with 100 epochs:

print("50 Epochs (Mean, Standard Deviation) :", mse_mean_norm, standard_dev_mse_norm)
print("100 Epochs (Mean, Standard Deviation) :", mse_mean_hundred, standard_dev_mse_hundred)
print(" The MSE for 100 epochs is much smaller than that of 50 epochs.")
print("The Standard Deviation for 100 epochs is also much smaller than that of 50 epochs.")




################## Part D #######################################3

# Using Part B Normalized data but increasing the number of hidden layers to 3 Hidden Layers
# Three hidden layers, each of 10 nodes and ReLU activation function.

def regression_model_three_hidden():
    """ A Function for the Regression Model.
    This is a Neural Network regression function with
    one hidden layer of 10 nodes,
    adam optimizer, and
    mean_squared_error loss function"""
    
    r_model = Sequential()
    r_model.add(Dense(10, activation ='relu', input_shape=(n_cols,)))
    
    # 3 Hidden Layers, 10 nodes, ReLu activation
    r_model.add(Dense(10, activation ='relu'))
    r_model.add(Dense(10, activation ='relu'))
    r_model.add(Dense(10, activation ='relu'))

    # Output layer
    r_model.add(Dense(1))
    
    # for training, using 'adam' optimizer
    r_model.compile(optimizer='adam', loss = 'mean_squared_error') 

    return r_model


# Repeat split, train, test 50 times, calculate MSE

times = 50
mse_list =[]  # for list of 50 Mean Squared Errors
    
for i in range(0, times):
        
        # split data
        x_train, x_test, y_train, y_test = train_test_split(normalized_features, target, test_size=0.30, random_state=10)
        
        # call the regression function to create model
        r_model = regression_model_three_hidden()
        
        # Fit the regression model: train the model
        r_model.fit(x_train, y_train,  epochs = 50, verbose=2)

        # make predictions using the trained model with the test set
        predictions = r_model.predict(x_test)
        
        # transform 'predictions' and 'y_test' to lists so can calculate mse
        prediction_list = predictions.tolist()
        y_test_list = y_test.tolist()

        # calculate the mean squared error
        mse = mean_squared_error(y_test_list,prediction_list)
        # append to a list
        mse_list.append(mse)
        
# Checking the length of the mse_list - should be 50

print("Length of the mse_list: ", len(mse_list))

# Report the mean and the standard deviation of the mean squared errors.
mse_mean_three = mean_s(mse_list)
standard_dev_mse_three = stdev_s(mse_list)

print("The Mean of the Mean Squared Error with 100 Epochs : ", mse_mean_three)
print("The Standard Deviation of the Mean Squared Errorfor Normalized data: ", standard_dev_mse_three)
print("Mean of the MSE with 1 hidden layer: ", mse_mean_norm)
print("Mean of the MSE with 3 hidden layers: ", mse_mean_three)
print("Standard Deviation of the MSE, 1 hidden layer: ", standard_dev_mse_norm)
print("Standard Deviation of the MSE, 3 hidden layers: ", standard_dev_mse_three)
print("Conclusion: Both the mean and standard deviation is reduced with the increase in hidden layers.")