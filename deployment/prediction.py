import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# Load All Files
with open('model.pkl', 'rb') as file_1:
  xgb_model = joblib.load(file_1)

def run():
    st.session_state.disabled = True
    booking_id = st.text_input('Booking ID', max_chars=8, key='booking_id')
    if re.match(r'INN[0-9]{5,5}',st.session_state.booking_id) is None:
       st.warning('The template of invoice is INN[0-9]{5}')
       st.session_state.disabled = True
    else:
      st.session_state.disabled = False
    booking_date = st.date_input('Booking Date', disabled=st.session_state.disabled, key='booking_date')
    ordered_date = st.date_input('Booking Date', disabled=st.session_state.disabled, 
                                 min_value=st.session_state.booking_date)
    repeated_guest = st.radio('Did they are reapeted guests?',('Yes','No'), index=1, disabled=st.session_state.disabled)
    if repeated_guest == 'Yes':
      repeated_guest = 1
      st.session_state.disable_repeated = False
    else:
      repeated_guest = 0
      st.session_state.disable_repeated = True
    no_of_previous_cancellations = st.slider('How Many times customer cancelled?', min_value=0, max_value=10, value=0, step=1, help='Cancel booking timed',
                                    disabled=st.session_state.disable_repeated)
    no_of_previous_bookings_not_canceled = st.slider('How Many times customer booked and no cancelled?', min_value=0, max_value=20, value=0, step=1, help='Booking timed',
                                    disabled=st.session_state.disable_repeated)
    with st.form('key=form_hotel_reservation'):
      no_of_adults = st.slider('How Many Adults?', min_value=1, max_value=10, value=1, step=1, help='Number of Adults',
                                      disabled=st.session_state.disabled)
      no_of_children = st.slider('How Many Children?', min_value=0, max_value=10, value=0, step=1, help='Number of Children',
                                      disabled=st.session_state.disabled)
      no_of_weekend_nights = st.slider('How Many Weekend Nights that customer stay?', min_value=0, max_value=8, value=0, step=1, help='Number of Weekend Nights',
                                      disabled=st.session_state.disabled)
      no_of_week_nights = st.slider('How Many Weekday Nights that customer stay?', min_value=0, max_value=20, value=0, step=1, help='Number of Weekday Nights',
                                      disabled=st.session_state.disabled)
      type_of_meal_plan = st.selectbox('What Meal Plan that customer want?',('Meal Plan 1','Meal Plan 2','Meal Plan 1','Not Selected'),
                                        index=0, disabled=st.session_state.disabled)
      required_car_parking_space = st.radio('Did they want parking space?',('Yes','No'), index=0, disabled=st.session_state.disabled)
      if required_car_parking_space == 'Yes':
        required_car_parking_space = 1
      else:
        required_car_parking_space = 0
      room_type_reserved = st.selectbox('What Type of room that customer want?',
                                        ('Room_Type 1','Room_Type 2','Room_Type 3','Room_Type 4','Room_Type 5','Room_Type 6','Room_Type 7'),
                                        index=0, disabled=st.session_state.disabled)
      market_segment_type = st.selectbox('What media that customer booked hotel from?',
                                        ('Online','Offline','Corporate','Complementary','Aviation'),
                                        index=0, disabled=st.session_state.disabled)
      avg_price_per_room = st.number_input('Average Price per Room',step=1.,format="%.2f", disabled=st.session_state.disabled)
      no_of_special_requests = st.slider('How Many Special Reqeust from this customer?', min_value=0, max_value=10, value=0, step=1, help='Special Requests',
                                      disabled=st.session_state.disabled)
      submitted = st.form_submit_button('Predict', disabled=st.session_state.disabled)

    # Create New Data

    data_inf = {
      'Booking_ID': booking_id,
      'no_of_adults': no_of_adults,
      'no_of_children': no_of_children,
      'no_of_weekend_nights': no_of_weekend_nights,
      'no_of_week_nights': no_of_week_nights,
      'type_of_meal_plan': type_of_meal_plan,
      'required_car_parking_space': required_car_parking_space,
      'room_type_reserved': room_type_reserved,
      'lead_time': int((ordered_date - booking_date).days),
      'arrival_year': ordered_date.year,
      'arrival_month': ordered_date.month,
      'arrival_date': ordered_date.day,
      'market_segment_type': market_segment_type,
      'repeated_guest': repeated_guest,
      'no_of_previous_cancellations': no_of_previous_cancellations,
      'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
      'avg_price_per_room': avg_price_per_room,
      'no_of_special_requests': no_of_special_requests
    }
    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submitted:
        y_pred_inf = xgb_model.predict(data_inf)
        if y_pred_inf[0] == 0:
            st.write('# This booking predicted to be not canceled')
        else:
            st.write('# This booking predicted to be canceled')