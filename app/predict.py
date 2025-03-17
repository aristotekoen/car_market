import pickle

import numpy as np
import pandas as pd
import streamlit as st
from catboost import Pool, CatBoostRegressor
from google.cloud import storage
from const import *
from reliability import reliability_score
from interpretability import pdp_cat, pdp_num
from io import BytesIO
import matplotlib.pyplot as plt

if "user_input" not in st.session_state:
    st.session_state['user_input'] = None
if "df_input" not in st.session_state:
    st.session_state['df_input'] = None
if "test_set" not in st.session_state:
    st.session_state['test_set'] = None
if "preds" not in st.session_state:
    st.session_state['preds'] = None
if "reliability" not in st.session_state:
    st.session_state['reliability'] = None
if "catboost_model_1" not in st.session_state:
    st.session_state['catboost_model_1'] = None
if "catboost_model" not in st.session_state:
    st.session_state['catboost_model'] = None
if "catboost_model_3" not in st.session_state:
    st.session_state['catboost_model_3'] = None

@st.cache_resource(show_spinner=False)
def load_model(name):
    client = storage.Client()
    bucket = client.get_bucket('price-estimation')
    blob = bucket.blob(f'models/catboost_{name}.cbm')
    model_name = f'catboost_{name}.cbm'
    blob.download_to_filename(model_name)
    cb = CatBoostRegressor().load_model(model_name)
    return cb

@st.cache_data(show_spinner=False)
def load_car_dictionnary():
    client = storage.Client()
    bucket = client.get_bucket('price-estimation')
    blob = bucket.blob('data/car_dictionary.pkl')
    dict_name = 'car_dictionary.pkl'
    blob.download_to_filename(dict_name)
    with open(dict_name, 'rb') as f:
        cars_dict = pickle.load(f)
    return cars_dict

@st.cache_data(show_spinner=False)
def load_test_set():
    client = storage.Client()
    bucket = client.get_bucket('price-estimation')
    blob_test = bucket.get_blob('data/train_test_sets/test_02032024.csv')
    test_bytes = blob_test.download_as_bytes()
    test = pd.read_csv(BytesIO(test_bytes), index_col=0)
    return test

def create_dict_for_pred(model):
    return {k: np.nan for k in model.feature_names_}

def predict_price(df, q1, q2, q3):
    price_q1 = q1.predict(df)[0]
    price_q2 = q2.predict(df)[0]
    price_q3 = q3.predict(df)[0]

    if price_q3 <= price_q1:
        return price_q3 , (price_q1 + price_q3)/2, price_q1
    if (price_q1 >= price_q2) or (price_q2 >= price_q3):
        return price_q1,  (price_q3 + price_q1)/2, price_q3


    return price_q1, price_q2, price_q3


cars_dict = load_car_dictionnary()

st.title(':car: AI powered Used Car Price Estimator')
st.markdown("""
### 🤖 How does this work?  
This **machine learning model** estimates the price of a used car based on various features such as **brand, model, mileage, registration year, engine, and more**.  

📊 The model was trained on real **used car listings from Greek websites**, so the predictions reflect the **Greek market**.  

✅ **Simply fill in the details** and click the **Predict** button to get:
- An **estimated price** for your car 💰
- A **price range** (low & high) 📈

Coming soon:
- A **confidence level** for accuracy 🎯
- Analysis of the impact of the main features on your car price
""")
st.header('📌 Car details')

brand = st.selectbox("Select the car brand", cars_dict.keys())
model = st.selectbox("Model", cars_dict[brand])
body_type = st.selectbox("Body type", body_types, help="If unknown leave as is")

st.header('ℹ️ Basic Information')

mileage = float(st.number_input("Mileage in km", value=10000))
registration_year = float(st.number_input("Registration year", min_value=2000, max_value=2025, format="%d"))
fuel_type = st.selectbox("Fuel type", fuel_types, index=0)
gearbox_type = st.selectbox("Gearbox type", gearbox_types, index=1)
seats = float(st.slider("Number of seats", min_value=1, max_value=10, format="%d", value=5))
doors = float(st.number_input("Number of doors", min_value=1, max_value=10, format="%d", value=5))
rim_size = float(st.number_input("Rim size (inches)?", min_value=10, max_value=30, help="Size of the wheel rims"))
drive_type = st.selectbox("Drive type", drive_types, help="FWD: Front Wheel Drive, RWD: Rear Wheel Drive, AWD: All Wheel Drive, 4WD: four wheel drive")
number_plate_ending = st.selectbox("Number of plate endings", number_plate_endings)

st.header('⚙️ Engine Information')

engine_size = float(st.number_input("Engine size in cc", value=1000))
engine_power = float(st.number_input("Engine power in bhp", value=100))


st.header('🔧 Vehicle condition ')
is_new = st.selectbox("Is your car new?", [True, False], index=1)
crashed = st.selectbox("Is the vehicle sold crashed?", [True, False], index=1)
never_crashed = st.selectbox("Was the vehicle crashed in the past?", [True, False], index=1)

st.header('🎨 Colors and interior material')
exterior_color = st.selectbox("Exterior color", exterior_colors)
interior_type = st.selectbox("Interior type", interior_types, help="If unknown leave as is")
interior_color = st.selectbox("Interior color", interior_colors, help="If unknown leave as is")
is_metallic = st.selectbox("Is your car paint color metallic?", [True, False], index=1)

st.header('✨ Extras')

extra_options = ['extra_turbo','extra_steering_lights','extra_collision_avoidance_system','extra_trip_computer', 'extra_wheelchair', 'extra_apple_carplay', 'extra_leather_seats', 'extra_anti_theft_system_gps', 'extra_rain_sensor', 'extra_panoramic_roof', 'extra_dvd', 'extra_automatic_parking', 'extra_multi_purpose_steering_wheel', 'extra_tcs_asr', 'extra_tv_camera',    'extra_power_windows', 'extra_bucket_seats', 'extra_hitch', 'extra_aircondition(a_c)',    'extra_heated_seats', 'extra_power_seats', 'extra_alumium_rims', 'extra_locking_differential', 'extra_android_auto', 'extra_roof_rails', 'extra_armored', 'extra_lane_assist',    'extra_immobilizer', 'extra_rear_view_camera', 'extra_air_suspension', 'extra_bluetooth','extra_parktronic', 'extra_isofix_children_seats', 'extra_electric_sunroof', 'extra_abs','extra_telephone', 'extra_led_lights', 'extra_gps', 'extra_automatic_air_conditioning','extra_central_locking', 'extra_hill_assist', 'extra_tft_screen', 'extra_cd_player','extra_head_up_display', 'extra_service_book', 'extra_power_mirrors', 'extra_eco_start_stop','extra_alarm', 'extra_xenon', 'extra_fog_lights', 'extra_power_steering', 'extra_radio_player','extra_usb', 'extra_cruise_control', 'extra_keyless', 'extra_esp']

# Générer dynamiquement les selectbox pour chaque option
with st.expander('✨ Select Extra Features', expanded=False):
    extras = {}
    for option in extra_options:
        label = option.replace("extra_", "").replace("_", " ").title()
        extras[option] = st.checkbox(label)

user_input = {'lat': lat,
              'lon': longitude,
              'is_new': is_new,
              'mileage': mileage,
              'crashed': crashed,
              'engine_size': engine_size,
              'registration_year': registration_year,
              'engine_power': engine_power,
              'fuel_type': fuel_type,
              'gearbox_type': gearbox_type,
              'brand': brand,
              'model': model,
              'never_crashed': never_crashed
              }

user_input.update(extras)

rest_dict = {
    'interior_type': interior_type,
    'seats': seats,
    'exterior_color': exterior_color,
    'number_plate_ending': number_plate_ending,
    'interior_color': interior_color,
    'rim_size': rim_size,
    'body_type': body_type,
    'drive_type': drive_type,
    'doors': doors,
    'is_metallic': is_metallic}

user_input.update(rest_dict)

st.session_state['user_input'] = None
st.session_state['user_input'] = user_input


st.header("🔮 Ready to estimate the price of your car?")

st.write("")
if st.button("Predict car price", use_container_width=True):
    st.session_state['preds'] = None
    st.session_state['reliability'] = None
    with st.spinner("Estimating car price..."):



        catboost_model = load_model('q2')
        catboost_model_1 = load_model('q1')
        catboost_model_3 = load_model('q3')

        st.session_state['catboost_model_1'] = catboost_model_1
        st.session_state['catboost_model'] = catboost_model
        st.session_state['catboost_model_3'] = catboost_model_3

        feature_names = catboost_model.feature_names_
        df_input = pd.DataFrame([user_input], columns=feature_names)

        st.session_state['df_input'] = df_input

        df_pool = Pool(df_input, cat_features=catboost_model.get_cat_feature_indices())

        price_q1, price_q2, price_q3 = predict_price(df_pool, catboost_model_1, catboost_model, catboost_model_3)
        st.session_state['preds'] = (price_q1, price_q2, price_q3)
        reliability = reliability_score(price_q1,price_q2,price_q3,0.2,20)
        st.session_state['reliability'] = reliability



if "preds" in st.session_state and st.session_state['preds'] is not None:
    preds = st.session_state['preds']
    price_q1, price_q2, price_q3 = preds[0], preds[1], preds[2]
    reliability = st.session_state['reliability']

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Estimated Low Price", value=f"{int(price_q1)} €")
    col2.metric(f"Estimated Price", value=f"{int(price_q2)} €")
    col3.metric(f"Estimated High Price", value=f"{int(price_q3)} €")

    if reliability >= 0.75:
        icon = "🟢"
    elif 0.25 <= reliability < 0.75:
        icon = "🟠"
    else:
        icon = "🔴"
    st.write("")
    st.metric(label="Reliability Score", value=f"{icon}  {int(reliability * 100)}%", delta="")

st.write("")
st.write("")
st.write("")

st.header('🔍 Effect of features on the estimated price')

st.markdown("""
In this section, we provide a visualisation of how a change on a certain feature of the car affects the estimated price while keeping all the other characteristics to the value you entered for estimating the price.   
  
""")

user_input = st.session_state['user_input']
list_features = [x for x in list(user_input.keys()) if x not in ['lat', 'lon', 'is_new', 'brand', 'model', 'never_crashed']]
user_input_effect = st.selectbox("Select one characteristic and see how it could affect the value of your car!",list_features)

if st.button('Show the effect!'):
    test_set = load_test_set()
    st.session_state['test_set'] = test_set

    test_set = st.session_state['test_set']
    df_input = pd.DataFrame([st.session_state['user_input']], columns=st.session_state['df_input'].columns)
    st.session_state['df_input'] = df_input

    catboost_model_1 = st.session_state['catboost_model_1']
    catboost_model_2 = st.session_state['catboost_model']
    catboost_model_3 = st.session_state['catboost_model_3']

    if isinstance(df_input[user_input_effect].values[0], float) or isinstance(df_input[user_input_effect].values[0],int):
        pdp_num(df_input, test_set, user_input_effect,[catboost_model_1,catboost_model_2,catboost_model_3])
        st.pyplot(plt.gcf())

    else:
        pdp_cat(df_input, test_set, user_input_effect, [catboost_model_1, catboost_model_2, catboost_model_3])
        st.pyplot(plt.gcf())