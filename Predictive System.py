import numpy as np
import pickle


#loading the saved Model
loaded_model = pickle.load(open('./trained_heart_model.sav', 'rb'))

input_data =(42,0,1,160,254,0,0,112,0,2.4,3,0,2)

#changing input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
  print('The Person is Healthy ')
else :
  print('The person is not Healthy ')
