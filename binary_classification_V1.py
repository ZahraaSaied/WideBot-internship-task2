# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:11:02 2020

@author: Zahraa
"""

# Import the necessary modules
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler



# Import the training_dataset
training_data = pd.read_csv("training.csv", sep = ";")
validation_data = pd.read_csv("validation.csv", sep = ";")
print(len(training_data)) # 3700
print(len(validation_data)) # 200

# Explore the data
print(training_data.head())
print(training_data.columns)
print(training_data.info())
print(training_data.describe())
print(training_data.dtypes)
print(training_data.classLabel.value_counts())

# Preprocessing the data

#### Handling format and data types ####

def format_data(data):
    # variable3
    data.variable2 = data.variable2.str.replace(",", ".")
    data.variable2 = data.variable2.astype(float)
    
    # variable3
    data.variable3 = data.variable3.str.replace(",", ".")
    data.variable3 = pd.to_numeric(data.variable3, errors='coerce')
    
    # variable8
    data.variable8 = data.variable8.str.replace(",", ".")
    data.variable8 = pd.to_numeric(data.variable8, errors='coerce')
    
    # Label
    data.classLabel = data.classLabel.str.replace(".", "")   


format_data(training_data)
format_data(validation_data)
print(training_data.head())
print(validation_data.head())

#### Handling missing values ####
print(training_data.isna().sum())

# Fit Imputer on trainng data only
only_numeric = training_data.select_dtypes(include=['float64', 'int64'])
print(only_numeric.isna().sum())
numeric_cols_with_nan = ['variable2', 'variable14', 'variable17']

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values = np.nan, strategy='mean')
imp.fit(training_data[numeric_cols_with_nan])

def handle_missing_values(data):
    # variable18 contais (60%) missing values, drop it
    data.drop("variable18", axis=1, inplace=True)
    
    # Variable 9, 10 and 12 contais the same categories (duplicate data)
    data.drop(["variable9", "variable10"], axis=1, inplace=True)
    
    # Transfrom data using fitted Imputer
    data[numeric_cols_with_nan] = imp.transform(data[numeric_cols_with_nan])

    # Impute categorical missing values
    # Fill with the most freq value (mode imputation)
    data.variable1 = data.variable1.fillna(data.variable1.value_counts().index[0])
    data.variable4 = data.variable4.fillna(data.variable4.value_counts().index[0])
    data.variable5 = data.variable5.fillna(data.variable5.value_counts().index[0])
    data.variable6 = data.variable6.fillna(data.variable6.value_counts().index[0])
    data.variable7 = data.variable7.fillna(data.variable7.value_counts().index[0])

handle_missing_values(training_data)
handle_missing_values(validation_data)

print(training_data.isna().sum())
print(validation_data.isna().sum())

# Incode categorical variables

cat_cols = ['variable1', 'variable4', 'variable5', 'variable6', 
                      'variable7', 'variable12', 'variable13', 'classLabel']
prefix_list = ['v1', 'v4', 'v5', 'v6', 'v7', 'v2', 'v13', 'Label']
    
training_data = pd.get_dummies(training_data, columns = cat_cols, drop_first = True, prefix = prefix_list)

validation_data = pd.get_dummies(validation_data, columns = cat_cols, drop_first = True, prefix = prefix_list)

print(training_data.info())
print(validation_data.info())


# Scale numeric columns
numeric_cols = ['variable2', 'variable3', 'variable8', 'variable11', 
                  'variable14', 'variable15', 'variable17', 'variable19']

scaler = StandardScaler()
scaler.fit(training_data[numeric_cols])
training_data[numeric_cols] = scaler.transform(training_data[numeric_cols])
validation_data[numeric_cols] = scaler.transform(validation_data[numeric_cols])

print(training_data[numeric_cols].describe())

## Issue => diffrent categores on the training and validtdion data