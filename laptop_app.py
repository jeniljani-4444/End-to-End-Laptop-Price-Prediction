import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import requests
import pickle

st.set_page_config(
    page_title="Laptop Price Prediction App",
    page_icon=":computer:",
    layout="wide"
)

# Custom CSS
st.markdown(
    """<style>
            footer {
            visibility: hidden;
      } </style>""",
    unsafe_allow_html=True
)

# Load data and models only once


@st.cache_resource()
def load_data_and_models():
    # Load data
    laptop = pd.read_pickle('laptop.pkl')

    # Load the ML model
    with open('pipe.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    return laptop, model


laptop, model = load_data_and_models()

# Lottie Animation


def lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Navigation
with st.sidebar:
    selected = option_menu(
        menu_title='Menu',
        menu_icon='menu-button-wide',
        default_index=0,
        options=['Description', 'Prediction', 'Analysis'],
        icons=['book', 'laptop', 'bar-chart'],
        orientation='vertical'
    )

# Description
if selected == 'Description':
    st.title("Laptop Price Prediction :computer:")
    st.subheader("Aim: The aim of the Laptop Price Prediction project is to develop an end-to-end machine learning model that can accurately predict the prices of laptops based on various features and specifications. This project leverages data scraped from Flipkart, a popular online shopping platform, to train and deploy a predictive model. The primary objectives of this project include:", divider='rainbow')

    left_col, right_col = st.columns(2)
    lottie = lottieurl(
        "https://lottie.host/a28e5c0f-c880-4118-86e4-704cd4a59ecc/D9Gn0j1sQX.json")

    with left_col:
        st.write("1) Data Collection: Gather laptop data from Flipkart, including information on laptop specifications such as brand, processor, RAM, storage, screen size, user ratings, and more. The aim is to create a comprehensive dataset that covers a wide range of laptop models.")
        st.write("2) Data Preprocessing: Clean and preprocess the collected data to handle missing values, outliers, and any inconsistencies. Ensure that the dataset is suitable for machine learning tasks.")
        st.write("3) Feature Engineering: Select relevant features and transform them to extract meaningful information that can aid in predicting laptop prices accurately. Feature engineering may involve text processing for product descriptions and one-hot encoding for categorical variables")
        st.write("4) Model Selection: Experiment with various machine learning algorithms and regression techniques to identify the most suitable model for price prediction. Consider algorithms like linear regression, decision trees, random forests, or advanced methods like gradient boosting and neural networks.")

    with right_col:
        st_lottie(lottie, height=500, key="laptop")

# Prediction
if selected == 'Prediction':
    st.title("Prediction")

    name = st.selectbox("Laptop Name", laptop['name'].unique())
    pros_name = st.selectbox("Processor Name", laptop['pros_name'].unique())
    pros_gen = st.selectbox("Processor Generation",
                            laptop['pros_gen'].unique())
    ram = st.selectbox("RAM", laptop['ram'].unique())
    ssd = st.selectbox("SSD", laptop['ssd'].unique())
    hdd = st.selectbox("HDD", laptop['hdd'].unique())
    graphic_card = st.selectbox(
        "Graphic Card", laptop['graphic_card'].unique())
    os = st.selectbox("OS", laptop['os'].unique())
    display = st.number_input("Display Size", min_value=11.6, max_value=17.3)
    touch = st.selectbox("Touchscreen", ['Yes', 'No'])
    touch = 1 if touch == 'Yes' else 0
    warranty = st.selectbox("Product Warranty", ['1 Year', '2 Year', '3 Year'])
    warranty = 1 if warranty == '1 Year' else (
        2 if warranty == '2 Year' else 3)
    rating = st.number_input("Product Rating", min_value=1.0, max_value=5.0)

    if st.button("Predict Price"):
        final_pred = [[name, pros_name, pros_gen, ram, ssd, hdd,
                       graphic_card, os, display, touch, warranty, rating]]
        predicted_price = int(np.exp(model.predict(final_pred)[0]))
        st.title(f"The predicted price of the laptop is: {predicted_price}")

# Analysis
if selected == 'Analysis':
    st.title("Analysis")
    st.subheader("Dataset")
    st.dataframe(laptop)
    # Shape of the dataset
    st.write("Number of rows: ", laptop.shape[0])
    st.write("Number of columns: ", laptop.shape[1])

    # Select category for Bar Chart
    st.subheader("Bar Chart")
    cols_name = st.selectbox("Select a category for Bar Chart", [
                             'name', 'pros_name', 'os'])

    # Create a Bar Chart

    name_by_price = laptop.groupby(by=[cols_name]).count()[
        ["price"]].sort_values("price")
    fig_name_prices = px.bar(
        name_by_price,
        x='price',
        y=name_by_price.index,
        template="plotly_white",
        orientation='h',
        height=500
    )
    fig_name_prices.update_xaxes(title_text="Counts")
    st.plotly_chart(fig_name_prices)

    # Donut Chart
    st.subheader("Donut Chart")
    input_donut_chart = st.multiselect("Select options for the Donut Chart:", [
                                       "ram", "graphic_card"], default=["ram"])

    def generate_donut_chart(column, title):
        counts = column.value_counts()
        labels = [f"{label} GB" for label in counts.index]
        fig = px.pie(
            names=labels,
            values=counts.values,
            title=title,
            hole=0.5
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig)

    # Generate Donut Chart for selected options
    if "ram" in input_donut_chart:
        generate_donut_chart(laptop['ram'], "RAM")

    if "graphic_card" in input_donut_chart:
        generate_donut_chart(laptop['graphic_card'], "Graphic Card")

    # Histogram
    st.subheader("Histogram")
    bins = st.slider("Select the number of bins",
                     min_value=5, max_value=20, value=10)

    # Create a Histogram
    fig = px.histogram(laptop['rating'], nbins=bins, title="Histogram")
    st.plotly_chart(fig)
