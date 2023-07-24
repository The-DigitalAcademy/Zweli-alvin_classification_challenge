import streamlit as st
import pandas as pd
import numpy as np
st.set_page_config(layout='wide',page_title='Address Book',page_icon='📚')

st.title('Uber pickups in NYC')


st.write('Select Model')
selected_option = st.selectbox("Select an option", ["Random Forest Cl"])
if selected_option == "Random Forest Cl":
            model_load_path = "model.pkl"
            with open(model_load_path, 'rb') as file:
                model = pickle.load(file)
                st.write("Machine learning Algorthm:", model)