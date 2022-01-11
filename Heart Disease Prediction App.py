#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 22:54:08 2022

@author: bunty
"""

import numpy as np 
import pickle            # for loading the saved model 
import streamlit as st   # for creating the web page

loaded_model = pickle.load(open('/Users/bunty/VS Code/MachineLearning/Heart-Disease-Prediction/trained_heart_model.sav', 'rb'))

def heart_disease_prediction(input_data):
    

    #changing input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance 
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0]==0):
      return 'The person is Healthy'
    else :
      return 'The Person is having Heart Disease'
      
      
     
def main():
    
    
    # giving a title
    st.title('Heart Disease Prediction App ')  
    
    
    # getting the input data from the user
    
    
    age = st.text_input('Age of a person')
    sex = st.text_input('Sex of a person 0->female, 1->male')
    chestpain = st.text_input('chest pain type')
    rbp = st.text_input('resting blood pressure')
    sc = st.text_input('serum cholestoral in mg/dl')
    fbs = st.text_input('fasting blood sugar &gt; 120 mg/dl 1 = true; 0 = false')
    restecg = st.text_input('resting electrocardiographic results in 0,1,2')
    maxheartrate = st.text_input('maximum heart rate achieved')
    exerciseinducedangina = st.text_input('exercise induced angina 1 = yes ; 0 = no')
    oldpeak = st.text_input('ST depression induced by exercise relative to rest')
    slope = st.text_input('the slope of the peak exercise ST segment')
    ca = st.text_input('number of major vessels 0-3 colored by flourosopy')
    thal = st.text_input('thal: 3 = normal; 6 = fixed defect; 7 = reversable defect')



    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction

    if st.button('Test Result'):
        diagnosis = heart_disease_prediction([age, sex, chestpain, rbp, sc, fbs, restecg , maxheartrate, exerciseinducedangina,oldpeak,slope,ca,thal])
    
    
    st.success(diagnosis)





if __name__ == '__main__': 
    main() 


















     