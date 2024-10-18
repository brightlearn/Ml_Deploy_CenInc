# -*- coding: utf-8 -*-
"""
Created on Wed Oct 9 15:21:03 2024

@author: 
"""

#  Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle

# Fetching UCI repositories
from ucimlrepo import fetch_ucirepo

# Fetch dataset and create dataframe
adult = fetch_ucirepo(id=2) 
adultDataFrame = adult.data.original

# Replace '?' with NaN
adultDataFrame.replace('?', np.nan, inplace=True)

# Clean values in income column
adultDataFrame['income'].replace('<=50K.', '<=50K', inplace=True)
adultDataFrame['income'].replace('>50K.', '>50K', inplace=True)

# Drop Duplicates
adultDataFrame = adultDataFrame.drop_duplicates()

# Clean education column
adultDataFrame['education_new'] = np.where(np.isin(adultDataFrame.education, ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th']),
                                           'Not HS-grad', adultDataFrame.education)

# Define bins and labels for age categories
bins = [0, 13, 20, 35, 61, 100]  # Age intervals
labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']  # Corresponding labels

# Categorize the age into groups
adultDataFrame['age_group'] = pd.cut(adultDataFrame['age'], bins=bins, labels=labels, right=False)

# Drop columns that are not needed
adultDataFrame.drop(columns=['fnlwgt', 'education', 'education-num', 'age'], inplace=True)

# Change workclass 'Never-worked' hours-per-week as 0
adultDataFrame.loc[adultDataFrame['workclass'] == 'Never-worked', 'hours-per-week'] = 0

# Cap 'hours-per-week' at 80
adultDataFrame['hours-per-week'] = adultDataFrame['hours-per-week'].apply(lambda x: 80 if x > 80 else x)


# Apply LabelEncoder to the columns that need to be encoded
# columns that we are encoding: 
columns_to_encode = ['age_group', 'marital-status', 'occupation', 'relationship','race', 
                     'sex', 'native-country', 'income' ,'education_new','workclass'] 
# Initialize the LabelEncoder
le = LabelEncoder()

# Dictionary to store mappings for each column
label_mappings = {}

# Apply label encoding and save mappings
for col in columns_to_encode:
    adultDataFrame[col] = le.fit_transform(adultDataFrame[col])
    label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# Display the mapping of each encoded column
for col, mapping in label_mappings.items():
    print(f"Mapping for {col}: {mapping}")
    
# Separate rows with and without missing values in 'occupation', 14 is mapping to null
df_missing =  adultDataFrame[adultDataFrame['occupation'] == 14]
df_not_missing =  adultDataFrame[adultDataFrame['occupation'] != 14]


# imputing missing values in occupation
# Define features and target
X =  df_not_missing.drop(columns=['occupation'])   # Features
y = df_not_missing['occupation'] # Target variable (occupation)


# Split the non-missing data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Use the trained model to predict the 'workclass' for rows with missing values
X_missing = df_missing.drop(columns=['occupation'])  # Features for rows with missing values
df_missing['occupation'] = rf.predict(X_missing)

# Combine the data back together
df_imputed = pd.concat([df_not_missing, df_missing])

# Display the imputed DataFrame
#print(df_imputed)

adultDataFrame=df_imputed.copy()


# Separate rows with and without missing values in 'occupation', 14 is mapping to null
df_missing =  adultDataFrame[adultDataFrame['workclass'] == 8]
df_not_missing =  adultDataFrame[adultDataFrame['workclass'] != 8]


# imputing missing values in occupation
# Define features and target
X =  df_not_missing.drop(columns=['workclass'])   # Features
y = df_not_missing['workclass'] # Target variable (occupation)


# Split the non-missing data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Use the trained model to predict the 'workclass' for rows with missing values
X_missing = df_missing.drop(columns=['workclass'])  # Features for rows with missing values
df_missing['workclass'] = rf.predict(X_missing)

# Combine the data back together
df_imputed = pd.concat([df_not_missing, df_missing])

# Display the imputed DataFrame
#print(df_imputed)

adultDataFrame=df_imputed.copy()

#Define X (features) and y (target)
X = adultDataFrame[['workclass', 'education_new', 'marital-status', 
                   'occupation', 'relationship', 'sex', 
                   'age_group', 'hours-per-week']]
# =============================================================================
# X = adultDataFrame[['education_new', 'marital-status', 
#                     'occupation', 'relationship', 'sex', 
#                     'age_group', 'hours-per-week']]
# =============================================================================
y = adultDataFrame['income']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE to the training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train a RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)


#Save the trained model to a pickle file
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
