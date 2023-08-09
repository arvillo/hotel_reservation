import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Pilih Page: ',('EDA','Predict Hotel Reservation Cancelation'))

if navigation == 'EDA':
    eda.run()
else:
    prediction.run()