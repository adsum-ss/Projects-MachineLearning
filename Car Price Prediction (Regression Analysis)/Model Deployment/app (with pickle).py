import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from PIL import Image
import pickle

st.set_page_config(
     page_title="Car Price Prediction",
     #page_icon="🧊",
     layout="centered",
     initial_sidebar_state="expanded"
 )

st.markdown('# <div style="text-align: center"> CAR PRICE PREDICTION APP </div>', unsafe_allow_html=True)

st.write("""
This app predicts the **Car Prices** of specific models!

You can find the process of the data preparation from the
[CarPricePrediction (EDA)](https://github.com/adsum-ss/Projects-DataAnalysisWithPython/tree/main/AutoScout%20EDA%20Project%20(CAPSTONE))
and model installation from the
[CarPricePrediction (Regression Analysis)](https://github.com/adsum-ss/Projects-MachineLearning/tree/main/Car%20Price%20Prediction%20(Regression%20Analysis))

""")

st.write('***')
col1, col2, col3 = st.columns(3)

img1 = Image.open("renault-logo.jpg")
col1.image(img1, width=202, output_format='auto')
    
img2 = Image.open("audi-logo.jpg")
col2.image(img2, width=204, output_format='auto')
    
img3 = Image.open("Opel-logo.jpg")
col3.image(img3, width=200, output_format='auto')
st.write('***')

img_car = Image.open("car.jpg")
st.sidebar.image(img_car, use_column_width=True, output_format='auto')
st.sidebar.header('Client Input Features')

def user_input_features():
    model = st.sidebar.selectbox('Model', 
                ('Audi A1','Audi A3','Opel Astra','Opel Corsa','Opel Insignia','Renault Clio','Renault Duster','Renault Espace'))
    gearing_type = st.sidebar.selectbox('Gearing Type', ('Manual', 'Automatic', 'Semi-automatic'))
    age = st.sidebar.selectbox('Age', (0, 1, 2, 3))
    km = st.sidebar.number_input('Km', 0, 317000, step=1000)
    hp_kw = st.sidebar.number_input('Hp (Kw)', 40, 239, 66, 20)
    data = {'Make_Model': model,
            'Hp_kW': hp_kw,
            'Km': km,
            'Age': age,
            'Gearing_Type': gearing_type}
    features = pd.DataFrame(data, index=[1])
    return features

input_df = user_input_features()
input_df[['Hp_kW','Km','Age']] = input_df[['Hp_kW','Km','Age']].astype(int)

st.subheader('Selected Features')
st.write(input_df)
      
enc = pickle.load(open('final_encoder', 'rb'))
model = pickle.load(open('RandomForest_final', 'rb'))

cat = input_df.select_dtypes("object").columns
input_df[cat] = enc.transform(input_df[cat])

pred = model.predict(input_df)

st.write('#')
if st.button("Predict"):
   st.code(f'$ {int(pred)}')
   
