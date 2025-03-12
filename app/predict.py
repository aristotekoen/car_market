import pandas as pd
from catboost import Pool, CatBoostRegressor
import numpy as np
import streamlit as st
import pickle
from google.cloud import storage
import os
from dotenv import load_dotenv
from const import *


def load_model():
    client = storage.Client()
    bucket = client.get_bucket('price-estimation')
    blob = bucket.blob('models/catboost_extras')
    model_name = 'catboost_extras'
    blob.download_to_filename(model_name)
    cb = CatBoostRegressor().load_model(model_name)
    return cb

def load_car_dictionnary():
    client = storage.Client()
    bucket = client.get_bucket('price-estimation')
    blob = bucket.blob('data/car_dictionary.pkl')
    dict_name = 'car_dictionary.pkl'
    blob.download_to_filename(dict_name)
    with open(dict_name, 'rb') as f:
        cars_dict = pickle.load(f)
    return cars_dict

def create_dict_for_pred(model):
    return {k: np.nan for k in model.feature_names_}

catboost_model = load_model()
feature_names = catboost_model.feature_names_
cars_dict = load_car_dictionnary()


mileage = st.number_input("Mileage in km", value=10000, format="%.2f")
engine_size = st.number_input("Engine size in cc", value=1000, format="%.2f")
registration_month = st.slider("Registration month", min_value=1, max_value=12, step=1)
registration_year = st.number_input("Registration year", min_value=1900, max_value=2025, format="%d")
engine_power = st.number_input("Engine power in bhp", value=100, format="%.2f")
seats = st.slider("Number of seats", min_value=1, max_value=10, format="%d", value=5)
doors = st.number_input("Number of doors", min_value=1, max_value=10, format="%d", value=5)
kteo_month = st.slider("KTEO month", step=1, min_value=1, max_value=12, format="%d", help="If not applicable leave as is")
kteo_year = st.slider("KTEO year", step=1,  min_value=1900, max_value=2025, format="%d", help="If not applicable leave as is")
co2_emissions = st.number_input("CO2 emissions (g/km)", value=100, format="%.2f")
battery_charge_time = st.slider("Battery charge time?", min_value=1, max_value=10, step=1, format="%d", help="If not applicable leave as is")
rim_size = st.number_input("Rim size (inches)?", min_value=10, max_value=30, format="%.2f", help="If  unknown leave empty, values should be between 10 and 30 inches")
vehicle_height = st.number_input("Vehicle height in mm", min_value= 1475, max_value=2500, format="%.2f", help="If unknown leave as is" )
number_of_gears = st.number_input("Number of gears?", min_value=2, max_value=10, step=1, format="%d", help="If unknown leave as is")
torque = st.number_input("Torque? (N.m)", min_value=70, max_value=900, format="%.2f", help="If unknown leave as is")
gross_weight = st.number_input("Gross Weight? (kg)", min_value=980, max_value=3500, format="%.2f", help="If unknown leave as is")
acceleration = st.number_input("Acceleration? (m/s^2)", min_value=1, max_value=40, step=1, format="%d", help="If unknown leave as is")
vehicle_width = st.number_input("Vehicle Width? (mm)", min_value=1000, max_value=3000, format="%.2f", help="If unknown leave as is")
vehicle_length = st.number_input("Vehicle length? (mm)", min_value=2500, max_value=7000, format="%.2f", help="If unknown leave as is")
top_speed = st.number_input("Top Speed? (Km/h)", min_value=100, max_value=400, format="%.2f", help="If unknown leave as is")
wheel_base = st.number_input("Wheelbase? (mm)", min_value=1500, max_value=4000, format="%.2f", help="If unknown leave as is")
fuel_consumption = st.number_input("Fuel consumption? L/100km", min_value=0, max_value=50, format="%.2f", help="If unknown leave as is")



#Cats
fuel_type = st.selectbox("Fuel type", fuel_types)
gearbox_type = st.selectbox("Gearbox type", gearbox_types)
brand = st.selectbox("Select the car brand", cars_dict.keys())
model = st.selectbox("Model", cars_dict[brand])
interior_type = st.selectbox("Interior type", interior_types, help="If unknown leave as is")
interior_color = st.selectbox("Interior color", interior_colors, help="If unknown leave as is")
body_type = st.selectbox("Body type", body_types, help="If unknown leave as is")
drive_type = st.selectbox("Drive type", drive_types, help="If unknown leave as is")


#Bools:
is_new = st.selectbox("Is your car new?", [True, False], index=1)
crashed = st.selectbox("Crashed?", [True, False], index=1)
is_metallic = st.selectbox("Is your car paint color metallic?", [True, False], index=1)

#Bools extras:
turbo = st.selectbox("Turbo?", [True, False])
steering_lights = st.selectbox("Steering lights?", [True, False])
collision_avoidance = st.selectbox("Collision avoidance?", [True, False])
exterior_color = st.selectbox("Exterior color", exterior_colors)
number_plate_ending = st.selectbox("Number of plate endings", number_plate_endings)

extra_options = ['extra_trip_computer', 'extra_wheelchair', 'extra_apple_carplay', 'extra_leather_seats', 'extra_anti_theft_system_gps', 'extra_rain_sensor', 'extra_panoramic_roof', 'extra_dvd', 'extra_automatic_parking', 'extra_multi_purpose_steering_wheel', 'extra_tcs_asr', 'extra_tv_camera',    'extra_power_windows', 'extra_bucket_seats', 'extra_hitch', 'extra_aircondition(a_c)',    'extra_heated_seats', 'extra_power_seats', 'extra_alumium_rims', 'extra_locking_differential', 'extra_android_auto', 'extra_roof_rails', 'extra_armored', 'extra_lane_assist', 'never_crashed',    'extra_immobilizer', 'extra_rear_view_camera', 'extra_air_suspension', 'extra_bluetooth','extra_parktronic', 'extra_isofix_children_seats', 'extra_electric_sunroof', 'extra_abs','extra_telephone', 'extra_led_lights', 'extra_gps', 'extra_automatic_air_conditioning','extra_central_locking', 'extra_hill_assist', 'extra_tft_screen', 'extra_cd_player','extra_head_up_display', 'extra_service_book', 'extra_power_mirrors', 'extra_eco_start_stop','extra_alarm', 'extra_xenon', 'extra_fog_lights', 'extra_power_steering', 'extra_radio_player','extra_usb', 'extra_cruise_control', 'extra_keyless', 'extra_esp']

# Générer dynamiquement les selectbox pour chaque option
extras = {}
for option in extra_options:
    label = option.replace("extra_", "").replace("_", " ").title()
    extras[option] = st.selectbox(f"{label}?", [True, False])


if st.button("Predict car price"):

    user_input = {'lat': lat,
 'lon': longitude,
 'is_new': is_new,
 'mileage': mileage,
 'crashed': crashed,
 'engine_size': engine_size,
 'registration_month': registration_month,
 'registration_year': registration_year,
 'engine_power': engine_power,
 'fuel_type': fuel_type,
 'gearbox_type': gearbox_type,
 'brand': brand,
 'model': model,
 'extra_turbo': turbo,
 'extra_steering_lights': steering_lights,
 'extra_collision_avoidance_system': collision_avoidance}

    user_input.update(extras)

    rest_dict = {
 'interior_type': interior_type,
 'seats': seats,
 'kteo': str(kteo_month) + "/" + str(kteo_year),
 'exterior_color': exterior_color,
 'number_plate_ending': number_plate_ending,
 'emissions_co2': co2_emissions,
 'battery_charge_time': battery_charge_time,
 'interior_color': interior_color,
 'rim_size': rim_size,
 'vehicle_height': vehicle_height,
 'number_of_gears': number_of_gears,
 'torque': torque,
 'gross_weight': gross_weight,
 'acceleration': acceleration,
 'vehicle_width': vehicle_width,
 'body_type': body_type,
 'vehicle_length': vehicle_length,
 'top_speed': top_speed,
 'wheelbase': wheel_base,
 'fuel_consumption': fuel_consumption,
 'drive_type': drive_type,
 'doors': doors,
 'is_metallic': is_metallic}

    user_input.update(rest_dict)
    df_input = pd.DataFrame([user_input], columns=feature_names)
    df_pool = Pool(df_input, cat_features=catboost_model.get_cat_feature_indices())
    price = catboost_model.predict(df_input)
    st.success(f"prix estime: {price}")