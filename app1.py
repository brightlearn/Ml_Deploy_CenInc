# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:28:21 2024

@author: shwet
"""

import numpy as np
import pickle
import streamlit as st
#import sklearn

# Load the model
model = pickle.load(open('random_forest_model.pkl', 'rb'))

# Define mappings
age_groupm = {'Adult': 0, 'Senior': 1, 'Teen': 2, 'Young Adult': 3} 
marital_statusm = {'Divorced': 0, 'Married-AF-spouse': 1, 'Married-civ-spouse': 2, 
                   'Married-spouse-absent': 3, 'Never-married': 4,
                   'Separated': 5, 'Widowed': 6}
occupation_mappingm = {'Adm-clerical': 0, 'Armed-Forces': 1, 'Craft-repair': 2, 
                       'Exec-managerial': 3, 'Farming-fishing': 4, 'Handlers-cleaners': 5,
                       'Machine-op-inspct': 6, 'Other-service': 7, 'Priv-house-serv': 8, 
                       'Prof-specialty': 9, 'Protective-serv': 10, 'Sales': 11, 
                       'Tech-support': 12, 'Transport-moving': 13, 'nan': 14}
relationshipm = {'Husband': 0, 'Not-in-family': 1, 'Other-relative': 2, 'Own-child': 3, 
                 'Unmarried': 4, 'Wife': 5}
sexm = {'Female': 0, 'Male': 1}
education_categorym = {'Assoc-acdm': 0, 'Assoc-voc': 1, 'Bachelors': 2, 'Doctorate': 3, 
                       'Elementary': 4, 'HS-grad': 5, 'Masters': 6, 'Middle School': 7, 
                       'Not HS-grad': 8, 'Prof-school': 9, 'Some-college': 10}
workclass_mappingm= {'Federal-gov': 0, 'Local-gov': 1, 'Never-worked': 2, 'Private': 3, 'Self-emp-inc': 4, 'Self-emp-not-inc': 5, 
             'State-gov': 6, 'Without-pay': 7}

# Function to categorize age based on predefined bins
def categorize_age(age):
    if 0 <= age <= 12:
        return 'Child'
    elif 13 <= age <= 19:
        return 'Teen'
    elif 20 <= age <= 34:
        return 'Young Adult'
    elif 35 <= age <= 60:
        return 'Adult'
    else:
        return 'Senior'

# Function to handle hours per week outliers
def outlier_hours_per_week(hr):
    return 80 if hr >= 80 else hr

# Streamlit UI
st.title("Income Prediction App")

# Collect user inputs through Streamlit widgets
age = st.number_input("Enter Age (17-99):", min_value=17, max_value=99, value=30)
age_group = categorize_age(age)
hours_per_week = st.number_input("Enter Hours per Week (0-99):", min_value=0, max_value=99, value=40)
hours_per_week = outlier_hours_per_week(hours_per_week)

marital_status = st.selectbox("Select Marital Status:", list(marital_statusm.keys()))
occupation = st.selectbox("Select Occupation:", list(occupation_mappingm.keys()))
workclass = st.selectbox("Select Workclass:", list(workclass_mappingm.keys()))
relationship = st.selectbox("Select Relationship Status:", list(relationshipm.keys()))
sex = st.selectbox("Select Gender:", list(sexm.keys()))
education_category = st.selectbox("Select Education Level:", list(education_categorym.keys()))

# Map inputs to encoded values
age_groupe = age_groupm.get(age_group)
marital_statuse = marital_statusm.get(marital_status)
occupatione = occupation_mappingm.get(occupation)
workclasse = workclass_mappingm.get(workclass)
relationshipe = relationshipm.get(relationship)
sexe = sexm.get(sex)
education_categorye = education_categorym.get(education_category)

# Prediction button
if st.button("Predict"):
    # Prepare final input array for the model
    final_features = np.array([hours_per_week, age_groupe, marital_statuse, 
                               occupatione,workclasse, relationshipe, education_categorye, sexe]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    
    # Display result
    if output == 0:
        st.success("Predicted Income: <=50K")
    else:
        st.success("Predicted Income: >50K")
