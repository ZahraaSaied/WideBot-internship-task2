
"""
Created on Wed Aug 12 13:11:02 2020

@author: Zahraa
"""

# Import the basic modules
import numpy as np
import pandas as pd

# Import the dataset

training_data = pd.read_csv("training.csv", sep = ";")
validation_data = pd.read_csv("validation.csv", sep = ";")
print(len(training_data)) # 3700
print(len(validation_data)) # 200

data = training_data.append(validation_data, ignore_index=True)
print(len(data))
print(data.index)

# Explore the data
print(data.head())
print(data.columns)
print(data.info())
print(data.describe())
print(data.dtypes)
print(data.classLabel.value_counts()) # imbalanced data

# Preprocessing the data

#### Handling format and data types ####

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


#### Handling missing values ###
print(data.isna().sum())

# variable18 contais (60%) missing values, drop it
data.drop("variable18", axis=1, inplace=True)

# Variable 9, 10 and 12 contais the same categories (duplicate data)
data.drop(["variable9", "variable10"], axis=1, inplace=True)

# Impute numeric missing values
only_numeric = data.select_dtypes(include=['float64', 'int64'])
print(only_numeric.isna().sum())
numeric_cols_with_nan = ['variable2', 'variable14', 'variable17']

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values = np.nan, strategy='mean')
imp.fit(data[numeric_cols_with_nan])
data[numeric_cols_with_nan] = imp.transform(data[numeric_cols_with_nan])

# Impute categorical missing values
# Fill with the most freq value (mode imputation)
data.variable1 = data.variable1.fillna(data.variable1.value_counts().index[0])
data.variable4 = data.variable4.fillna(data.variable4.value_counts().index[0])
data.variable5 = data.variable5.fillna(data.variable5.value_counts().index[0])
data.variable6 = data.variable6.fillna(data.variable6.value_counts().index[0])
data.variable7 = data.variable7.fillna(data.variable7.value_counts().index[0])

print(data.isna().sum())
print(data.info())

# Incode categorical variables

cat_cols = ['variable1', 'variable4', 'variable5', 'variable6', 
                  'variable7', 'variable12', 'variable13', 'classLabel']
prefix_list = ['v1', 'v4', 'v5', 'v6', 'v7', 'v2', 'v13', 'Label']

data = pd.get_dummies(data, columns=cat_cols, drop_first=True, prefix=prefix_list)
print(data.info())

#### Splitting data and get the original datasets #####

features = data.iloc[:,:-1].values
target = data.iloc[:, -1].values

X_train = features[:3700, :]
X_test = features[3700:, :]
y_train = target[:3700]
y_test = target[3700:]


# Scale numeric columns: index 0 => 7
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train[:, 0:7])
X_train[:, 0:7] = scaler.transform(X_train[:, 0:7])
X_test[:, 0:7] = scaler.transform(X_test[:, 0:7])

# Applying PCA for dimentionaliy reduction
from sklearn.decomposition import PCA
pca = PCA(n_components = 10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print(pca.explained_variance_ratio_.cumsum())

# Applying Gradient boosting for the classification task
import xgboost as xgb
classifier = xgb.XGBClassifier(objective='binary:logistic', n_estimators=15, seed=123)
classifier.fit(X_train, y_train)

# Predict the test set resluts
y_pred = classifier.predict(X_test)

####  Validation of model ####

# Training score
print("Training score: ",classifier.score(X_train, y_train))

# Test score
print("Test score: ",classifier.score(X_test, y_test))

# The confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix: ")
print(cm)

cl_report = classification_report(y_test, y_pred)
print("Classification report: ")
print(cl_report)

auc = roc_auc_score(y_test, y_pred)
print("AUC: ", auc)
