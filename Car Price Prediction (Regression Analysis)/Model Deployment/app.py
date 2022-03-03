import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from PIL import Image

st.set_page_config(
     page_title="Car Price Prediction",
     page_icon="🧊",
     layout="centered",
     #initial_sidebar_state="expanded"
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

with col1:
    img3 = Image.open("renault-logo.jpg")
    st.image(img3, width=202, output_format='auto')
    
with col2:
    img1 = Image.open("audi-logo.jpg")
    st.image(img1, width=204, output_format='auto')
    
with col3:
    img2 = Image.open("Opel-logo.jpg")
    st.image(img2, width=200, output_format='auto')
st.write('***')

img_car = Image.open("car.jpg")
st.sidebar.image(img_car, width=300, output_format='auto')
st.sidebar.header('Client Input Features')

def user_input_features():
    model = st.sidebar.selectbox('Model', 
                ('Audi A1','Audi A3','Opel Astra','Opel Corsa','Opel Insignia','Renault Clio','Renault Duster','Renault Espace'))
    gearing_type = st.sidebar.selectbox('Gearing Type', ('Manual', 'Automatic', 'Semi-automatic'))
    age = st.sidebar.selectbox('Age', (0, 1, 2, 3))
    km = st.sidebar.number_input('Km', 0, 317000, 1000)
    hp_kw = st.sidebar.number_input('Hp (Kw)', 40, 239, 50)
    data = {'Make_Model': model,
            'Hp_kW': hp_kw,
            'Km': km,
            'Age': age,
            'Gearing_Type': gearing_type}
    features = pd.DataFrame(data, index=[1])
    return features

input_df = user_input_features()

st.subheader('Selected Features')
#st.write(input_df)

def inner_func(enc, model):
    cat2 = input_df.select_dtypes("object").columns
    input_df[cat2] = enc.transform(input_df[cat2])
    pred = model.predict(input_df)
    return pred
    
@st.cache(suppress_st_warning=True)
def model_implement():
    
    df = pd.read_csv("scoutcar_simplified.csv")
    
    X= df.drop("Price", axis=1)
    y= df["Price"]
    cat = X.select_dtypes('object').columns
    enc = OrdinalEncoder()
    enc.fit(X[cat])
    X[cat] = enc.transform(X[cat])

    model = RandomForestRegressor(max_depth=14, min_samples_split=10, n_estimators=180, random_state=101).fit(X, y)
    return inner_func(enc, model)

st.write('#')
if st.button("Predict"):
   pred = model_implement()
   st.error(int(pred))
   
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: brown;
    font-size: 20px;
    height: 2em;
    width: 6em;
    border-radius: 10px 10px 10px 10px;
}
</style>""", unsafe_allow_html=True)

