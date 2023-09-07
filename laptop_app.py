import streamlit as st
import pickle
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import plotly.express as px
from streamlit_lottie import st_lottie
import requests


st.set_page_config(page_title="Laptop App",
                   page_icon=":computer:", layout="wide")
# Custom CSS
st.markdown(

    """<style>
            footer {
            visibility: hidden;
      } """,
    unsafe_allow_html=True
)



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

    st.subheader("Aim: The aim of the Laptop Price Prediction project is to develop an end-to-end machine learning model that can accurately predict the prices of laptops based on various features and specifications. This project leverages data scraped from Flipkart, a popular online shopping platform, to train and deploy a predictive model. The primary objectives of this project include:", divider="rainbow")

    left_col, right_col = st.columns(2)
    lottie = lottieurl("https://lottie.host/a28e5c0f-c880-4118-86e4-704cd4a59ecc/D9Gn0j1sQX.json")

    with left_col:
        st.write("1) Data Collection: Gather laptop data from Flipkart, including information on laptop specifications such as brand, processor, RAM, storage, screen size, user ratings, and more. The aim is to create a comprehensive dataset that covers a wide range of laptop models.")

        st.write("2) Data Preprocessing: Clean and preprocess the collected data to handle missing values, outliers, and any inconsistencies. Ensure that the dataset is suitable for machine learning tasks.")

        st.write("3) Feature Engineering: Select relevant features and transform them to extract meaningful information that can aid in predicting laptop prices accurately. Feature engineering may involve text processing for product descriptions and one-hot encoding for categorical variables")

        st.write("4) Model Selection: Experiment with various machine learning algorithms and regression techniques to identify the most suitable model for price prediction. Consider algorithms like linear regression, decision trees, random forests, or advanced methods like gradient boosting and neural networks.")

        st.write("5) Evaluation: Assess the model's performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2) to measure how well it predicts laptop prices.")

        st.write("6) Deployment: Create a user-friendly interface or application that allows users to input laptop specifications and receive real-time price predictions. The deployment can be done using web frameworks or mobile applications.")

    with right_col:
        st_lottie(lottie, height=500, key="laptop")

# Prediction
if selected == 'Prediction':
    @st.cache_resource(experimental_allow_widgets=True)
    def predict_func():
        st.title("Prediction")

        pickle_in = open('pipe.pkl', 'rb')
        pipe = pickle.load(pickle_in)
        df_pickle_in = open('laptop.pkl', 'rb')
        laptop = pickle.load(df_pickle_in)

        name = st.selectbox("Laptop Name", laptop['name'].unique())

        pros_name = st.selectbox(
            "Processor Name", laptop['pros_name'].unique())

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

    predict_func()

st.set_option('deprecation.showPyplotGlobalUse', False)

# Analysis
if selected == 'Analysis':
    @st.cache_resource(experimental_allow_widgets=True)
    def analysis():
        pickle_in = open('pipe.pkl', 'rb')
        pipe = pickle.load(pickle_in)
        df_pickle_in = open('laptop.pkl', 'rb')
        laptop = pickle.load(df_pickle_in)

        # Bar Chart
        st.title("Bar Chart")
        cols_name = st.selectbox("Select a category", [
                                 'name', 'pros_name', 'os'])

        name_by_price = (
            laptop.groupby(by=[cols_name]).count()[
                ["price"]].sort_values("price")
        )

        fig_name_prices = px.bar(
            name_by_price,
            x='price',
            y=name_by_price.index,
            template="plotly_white",
            orientation='h',
            height=500,

        )

        fig_name_prices.update_xaxes(title_text="Counts")
        st.plotly_chart(fig_name_prices)

        # Line Chart
        st.title("Line Chart")
        line_cols = st.selectbox("Select a category", ['ram', 'graphic_card'])
        line_by_price = (
            laptop.groupby(by=[line_cols]).count()[
                ["price"]].sort_values("price")
        )

        fig = px.line(line_by_price, x=line_by_price.index,
                      y='price')
        st.plotly_chart(fig)

        # Donut Chart
        st.title("Donut Chart")
        input_donut_chart = st.multiselect(
            "Select options:",
            ["ram", "graphic_card"],
            default="ram"
        )

        def generate_donut_chart(selected_option, column, title):
            counts = column.value_counts()
            labels = [f"{label} GB" for label in counts.index]
            fig = px.pie(
                names=labels,
                values=counts.values,
                title=title,
                hole=0.5

            )
            return fig

        if "ram" in input_donut_chart:
            fig1_py = generate_donut_chart("ram", laptop['ram'], "RAM")
            fig1_py.update_traces(textposition='inside',
                                  textinfo='percent+label')
            st.plotly_chart(fig1_py)

        if "graphic_card" in input_donut_chart:
            fig2_py = generate_donut_chart(
                "graphic_card", laptop['graphic_card'], "Graphic Card")
            fig2_py.update_traces(textposition='inside',
                                  textinfo='percent+label')
            st.plotly_chart(fig2_py)

        # Histogram
        st.title("Histogram")

        # Add a sidebar for customization
        bins = st.slider("Select the number of bins",
                         min_value=5, max_value=20, value=10)

        # Create a histogram using Plotly Express
        fig = px.histogram(laptop['rating'], nbins=bins, title="Histogram")
        st.plotly_chart(fig)

    analysis()


