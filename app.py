import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.title('Fuel Efficiency Prediction')

with open(r'C:\Users\ricar\OneDrive\Documents\4Geeks_Projects\Fuel_Efficiency_Project\models\finalized_xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

drive_mapping = {
    '2 Wheel Drive': 7,
    '4 Wheel Drive': 8,
    '4WD and AWD': 9,
    'All Wheel Drive': 10,
    'Front Wheel Drive': 11,
    'Part time 4WD': 12,
    'Rear Wheel Drive': 13,
    'Unknown': 14}

vclass_mapping = {
    'Compact Cars': 15,
    'Large Cars': 16,
    'Midsize Cars': 17,
    'Midsize Station Wagons': 18,
    'Midsize-Large Station Wagons': 19,
    'Minicompact Cars': 20,
    'Minivan - 2WD': 21,
    'Minivan - 4WD': 22,
    'Small Pickup Trucks': 23,
    'Small Pickup Trucks 2WD': 24,
    'Small Pickup Trucks 4WD': 25,
    'Small Sport Utility Vehicle 2WD': 26,
    'Small Sport Utility Vehicle 4WD': 27,
    'Small Station Wagons': 28,
    'Special Purpose Vehicle': 29,
    'Special Purpose Vehicle 2WD': 30,
    'Special Purpose Vehicle 4WD': 31,
    'Special Purpose Vehicles': 32,
    'Special Purpose Vehicles/2wd': 33,
    'Special Purpose Vehicles/4wd': 34,
    'Sport Utility Vehicle - 2WD': 35,
    'Sport Utility Vehicle - 4WD': 36,
    'Standard Pickup Trucks': 37,
    'Standard Pickup Trucks 2WD': 38,
    'Standard Pickup Trucks 4WD': 39,
    'Standard Pickup Trucks/2wd': 40,
    'Standard Sport Utility Vehicle 2WD': 41,
    'Standard Sport Utility Vehicle 4WD': 42,
    'Subcompact Cars': 43,
    'Two Seaters': 44,
    'Vans': 45,
    'Vans Passenger': 46,
    'Vans, Cargo Type': 47,
    'Vans, Passenger Type': 48
}

fuelType_mapping = {
    'Diesel': 49,
    'Midgrade': 50,
    'Premium': 51,
    'Regular': 52,
}

trany_mapping = {
    'Automatic': 53,
    'Manual': 54,
}

def predict_mpg(input_data):
    return model.predict(np.array(input_data).reshape(1, -1))


# User inputs
cylinders = st.slider('Cylinders', min_value=2, max_value=16, value=4)
displ = st.slider('Displacement', min_value=0.0, max_value=10.0, value=2.0, step=0.1, format="%.1f")
drivetrain = st.selectbox('Drivetrain', ['2 Wheel Drive','4 Wheel Drive','4WD and AWD','All Wheel Drive','Front Wheel Drive','Part time 4WD','Rear Wheel Drive','Unknown'])
vehicle_class = st.selectbox('Vehicle Class', ['Compact Cars', 'Subcompact Cars','Midsize Cars', 'Large Cars', 'Two Seaters', 'Standard Pickup Trucks',
                                                'Sport Utility Vehicle - 4WD', 'Small Station Wagons', 'Small Sport Utility Vehicle 4WD',
                                                'Sport Utility Vehicle - 2WD', 'Minicompact Cars', 'Special Purpose Vehicles', 
                                               'Standard Pickup Trucks 2WD', 'Standard Pickup Trucks 4WD', 'Standard Sport Utility Vehicle 4WD',
                                                'Vans', 'Small Sport Utility Vehicle 2WD', 'Special Purpose Vehicle 2WD',
                                               'Midsize-Large Station Wagons', 'Midsize Station Wagons', 'Small Pickup Trucks',
                                               'Small Pickup Trucks 2WD', 'Standard Sport Utility Vehicle 2WD', 'Vans, Cargo Type',
                                               'Minivan - 2WD', 'Special Purpose Vehicle 4WD', 'Vans, Passenger Type',
                                               'Small Pickup Trucks 4WD', 'Minivan - 4WD', 'Standard Pickup Trucks/2wd', 'Vans Passenger', 
                                               'Special Purpose Vehicles/2wd', 'Special Purpose Vehicles/4wd', 'Special Purpose Vehicle'])
transmission = st.selectbox('Transmission', ['Automatic', 'Manual'])
fuel_type = st.selectbox('Fuel Type', ['Diesel', 'Midgrade', 'Premium', 'Regular'])
tCharger = st.radio('Turbocharged', ['Yes', 'No'])
sCharger = st.radio('Supercharged', ['Yes', 'No'])
startStop = st.radio('Start-Stop System', ['Yes', 'No'])
year = st.slider('Year', min_value=1984, max_value=2024, value=2000)
guzzler = st.radio('Guzzler', ['Yes', 'No'])

num_features = 55
feature_vector = np.zeros(num_features)


feature_vector[0] = cylinders
feature_vector[1] = displ
feature_vector[2] = 1 if tCharger == 'Yes' else 0
feature_vector[3] = 1 if sCharger == 'Yes' else 0
feature_vector[4] = 1 if startStop == 'Yes' else 0
feature_vector[5] = year
feature_vector[6] = 1 if guzzler == 'Yes' else 0

feature_vector[drive_mapping[drivetrain]] = 1
feature_vector[vclass_mapping[vehicle_class]] = 1
feature_vector[fuelType_mapping[fuel_type]] = 1
feature_vector[trany_mapping[transmission]] = 1

if st.button('Predict MPG'):
    prediction = predict_mpg(feature_vector)
    st.write(f'Estimated Combined MPG: {prediction[0]:.2f}')

