import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('diabetes.sav', 'rb'))
def prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    Pregnancies=st.text_input("Number of Pregnancies")
    Glucose=st.text_input("Glucose Level")
    BloodPressure=st.text_input("Blood Pressure Value")
    SkinThickness=st.text_input("Skin Thickness Level")
    Insulin=st.text_input("Insulin Level")
    BMI=st.text_input("BMI value")
    DiabetesPedigreeFunction=st.text_input("Diabetes Pedigree Function av")
    Age=st.text_input("Age of the Person")
    final_Prediction = ''
    if st.button("Diabetes Prediction Result"):
          final_Prediction=prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
          st.success(final_Prediction)

if __name__=="__main__":
    main()


print(prediction([0,137,40,35,1684,3.1,2.288,33]))


