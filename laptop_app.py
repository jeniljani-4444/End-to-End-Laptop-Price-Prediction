import streamlit as st
import pickle
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go

# Custom CSS
st.markdown(

    """<style>
            footer {
            visibility: hidden;
      } """,
    unsafe_allow_html=True
)


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
if selected == 'Descripiton':
    pass


# Prediction
if selected == 'Prediction':
    @st.cache_resource(experimental_allow_widgets=True)
    def predict_func():
        st.title("Laptop Price Prediction :computer:")

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
                fig1_py.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig1_py)

        if "graphic_card" in input_donut_chart:
                fig2_py = generate_donut_chart(
                    "graphic_card", laptop['graphic_card'], "Graphic Card")
                fig2_py.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig2_py)

        # Histogram
        st.title("Histogram")

        # Add a sidebar for customization
        bins = st.slider("Select the number of bins", min_value=5, max_value=20, value=10)

        # Create a histogram using Plotly Express
        fig = px.histogram(laptop['rating'], nbins=bins, title="Histogram")
        st.plotly_chart(fig)
        
    analysis()
