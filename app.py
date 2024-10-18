import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle


app = Flask(__name__)
model = pickle.load(open('random_forest_model.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')

age_groupm={'Adult': 0, 'Senior': 1, 'Teen': 2, 'Young Adult': 3} 
marital_statusm={'Divorced': 0, 'Married-AF-spouse': 1, 'Married-civ-spouse': 2, 
                  'Married-spouse-absent': 3, 'Never-married': 4,
                  'Separated': 5, 'Widowed': 6}
occupation_mappingm={'Adm-clerical': 0, 'Armed-Forces': 1, 'Craft-repair': 2, 
              'Exec-managerial': 3, 'Farming-fishing': 4, 'Handlers-cleaners': 5,
              'Machine-op-inspct': 6, 'Other-service': 7, 'Priv-house-serv': 8, 
              'Prof-specialty': 9, 'Protective-serv': 10, 'Sales': 11, 
              'Tech-support': 12, 'Transport-moving': 13, 'nan': 14}
relationshipm= {'Husband': 0, 'Not-in-family': 1, 'Other-relative': 2, 'Own-child': 3, 
                'Unmarried': 4, 'Wife': 5}
sexm={'Female': 0, 'Male': 1}
education_categorym= {'Assoc-acdm': 0, 'Assoc-voc': 1, 'Bachelors': 2, 'Doctorate': 3, 'Elementary': 4, 'HS-grad': 5, 'Masters': 6, 'Middle School': 7, 'Not HS-grad': 8, 'Prof-school': 9, 'Some-college': 10}
workclass_mappingm= {'Federal-gov': 0, 'Local-gov': 1, 'Never-worked': 2, 'Private': 3, 'Self-emp-inc': 4, 'Self-emp-not-inc': 5, 
             'State-gov': 6, 'Without-pay': 7}

# Function to categorize age based on predefined bins
def categorize_age(age):
    if 0  <= age <= 12:
        return 'Child'
    elif 13 <= age <= 19:
        return 'Teen'
    elif 20 <= age <=34:
        return 'Young Adult'
    elif 35 <= age <=60:
        return 'Adult'
    else:
        return 'Senior'  # Handle cases where age might be out of expected range
    
    
    
    
# Function to categorize grade levels
def categorize_grade(grade):
#    if grade in ['Preschool', '1st-4th']:
#       return 'Elementary'
#   elif grade in ['5th-6th', '7th-8th']:
#        return 'Middle School'
#   elif grade in ['9th', '10th', '11th', '12th']:
#        return 'Not HS-grad'
#    else:
    if grade in ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th']:
        return 'Not HS-grad'
    else:
        return grade

    
# Function to categorize grade levels
def outlier_hours_per_week(hr):
#replacing the outliers in hours per week column:  upperbound outliers with 80
    if hr >= 80:
        return 80
    else:  
        return hr

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
  #  int_features = [int(x) for x in request.form.values()]
       # Get data from form (the original values)
    hours_per_week = int(request.form['hours_per_week'])
    age_group = int(request.form['age_group'])
    
    
    
    
    # Handling outlier in hours_per_week 
    hours_per_week = outlier_hours_per_week(hours_per_week)
    sex = (request.form['sex'])
      # Categorize the age 
    age_group = categorize_age(age_group)
        
    marital_status = request.form['marital-status']
    occupation = request.form['occupation']
    workclass = request.form['workclass']
    relationship = request.form['relationship']
    
    education_category = request.form['education_new']
       # Categorize the education_category 
    education_category = categorize_grade(education_category)
    
    
    # Convert original values to encoded values (similar to previous steps)
    age_groupe = age_groupm[age_group]  
    marital_statuse = marital_statusm[marital_status]
    occupatione = occupation_mappingm[occupation]
    workclasse = workclass_mappingm[workclass]
    relationshipe = relationshipm[relationship]
    education_categorye = education_categorym[education_category]   
    sexe = sexm[sex]  
       
    final_features = np.array([hours_per_week,
                               age_groupe,
                               marital_statuse,
                               occupatione,
                               workclasse,
                                 relationshipe,
                                 education_categorye,sexe]).reshape(1,-1)
    
    
    final_features = np.array([marital_statuse, occupatione,workclasse,relationshipe,
                               hours_per_week,
          sexe, age_groupe, education_categorye]).reshape(1,-1)
  
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if output == 0:
        output_1="<=50K"
    else:
        output_1=">50K"
    return render_template('index.html', prediction_text='Income Salary is {}'.format(output_1))


if __name__ == "__main__":
    app.run(debug=True)