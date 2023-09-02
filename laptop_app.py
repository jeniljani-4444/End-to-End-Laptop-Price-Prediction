import streamlit as st
import pickle
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu


st.markdown(

    """<style>
            footer {
            visibility: hidden;
            }
    </style>
    """,
    unsafe_allow_html=True
)

# Navigation
with st.sidebar:
    selected = option_menu(
        menu_title='Menu',
        menu_icon='menu-button-wide',
        default_index=0,
        options=['Description', 'Prediction'],
        icons=['book', 'laptop'],
        orientation='vertical'
    )

if selected == 'Descripiton':
    pass



def predict_func():
    if selected == 'Prediction':
        st.title("Laptop Price Prediction :computer:")

        pickle_in = open('pipe.pkl', 'rb')
        pipe = pickle.load(pickle_in)
        df_pickle_in = open('laptop.pkl', 'rb')
        laptop = pickle.load(df_pickle_in)

        name = st.selectbox("Laptop Name", laptop['name'].unique())

        pros_name = st.selectbox("Processor Name", laptop['pros_name'].unique())

        pros_gen = st.selectbox("Processor Genereation",
                                laptop['pros_gen'].unique())

        ram = st.selectbox("RAM", laptop['ram'].unique())

        ssd = st.selectbox("SSD", laptop['ssd'].unique())

        hdd = st.selectbox("HDD", laptop['hdd'].unique())

        graphic_card = st.selectbox(
            "Graphic Card", laptop['graphic_card'].unique())

        os = st.selectbox("OS", laptop['os'].unique())

        display = st.number_input(
            "Display Size", min_value=11.6, max_value=17.3)

        touch = st.selectbox("Touchscreen", ['Yes', 'No'])

        if touch == 'Yes':
            touch = 1
        else:
            touch = 0

        warranty = st.selectbox("Product Warranty", [
            '1 Year', '2 Year', '3 Year'])

        if warranty == '1 Year':
            warranty = 1
        elif warranty == '2 Year':
            warranty = 2
        else:
            warranty = 3

        rating = st.number_input(
            "Product Rating", min_value=1.0, max_value=5.0)

        if st.button("Predict Price"):

            final_pred = [[name, pros_name, pros_gen, ram, ssd, hdd, graphic_card, os, display, touch,
                        warranty, rating]]

            st.title("The predicted price of lapotop is: " +
                    str(int(np.exp(pipe.predict(final_pred)[0]))))


if __name__ == '__main__':
    predict_func()
