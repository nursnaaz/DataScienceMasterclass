import streamlit as st
import joblib
import numpy as np
import pandas as pd
from preprocessing import CustomTransformer

model = joblib.load('preprocess_model.pkl')

st.title("Car Price Prediction - Toy Example")
name = st.selectbox(
    'Select the Brand of the Car?',
    ('maruti','honda','toyota'))
fuel = st.selectbox(
    'Select the Fuel Type of the Car?',
    ('petrol','diesel'))
km = st.number_input('Select the Km_Driven by the vehicle')

def predict_price(model, name, fuel, km):
    new_test = [name,fuel,km]
    new_ = pd.DataFrame(new_test).T
    new_.columns = ['name','fuel_type', 'km_driven']
    return model.predict(new_)[0]    

if st.button('Predict'):
    result_petrol  = predict_price(model, name, 'petrol', km)
    result_diesel  = predict_price(model, name, 'diesel', km)
    st.write("Predicted ", name ," Car Price with fuel comparison: ")
    st.write("Predicted  ", name ," Car Price with Petrol : ",result_petrol)
    st.write("Predicted  ", name ," Car Price with Diesel : ",result_diesel)
    


