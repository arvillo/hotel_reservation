import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
from sklearn.preprocessing import OrdinalEncoder

st.set_page_config(
    page_title='Hotel Reservation - EDA',
    layout='wide',
    initial_sidebar_state='expanded'
)

def run():
    cat_col = ['type_of_meal_plan','required_car_parking_space','room_type_reserved','market_segment_type','repeated_guest']
    num_col = ['no_of_adults','no_of_children','no_of_weekend_nights','no_of_week_nights','lead_time',
        'arrival_year','arrival_month','arrival_date','no_of_previous_cancellations','no_of_previous_bookings_not_canceled',
        'avg_price_per_room','no_of_special_requests']
    # Title page
    st.title('Hotel Reservation Cancelation Prediction')

    # Markdown 
    st.markdown('---')
    data = pd.read_csv('Hotel Reservations.csv')

    # Bar Chart untuk Meal Plan
    st.write('#### Bar Chart of Meal Plan')
    fig = plt.figure(figsize=(15,5))
    data.groupby('type_of_meal_plan')['Booking_ID'].count().sort_values(ascending=False).plot.bar(x='type_of_meal_plan',
        y='booking_status', figsize=(10,10), title='What kind of meal plan the customer want?')
    st.pyplot(fig)
    st.write('Kebanyakan Customer memilih tipe makanan Meal Plan 1 untuk makanan mereka di hotel Liburan Skuy')
    st.markdown('---')

    # Pie Chart untuk Parking Space
    st.write('#### Pie Chart for Parking Space')
    fig = plt.figure(figsize=(15,5))
    data.groupby('required_car_parking_space')['Booking_ID'].count().plot.pie(y='Booking_ID', autopct='%0.2f', figsize=(10,10), 
        title='Is Customer need parking space?', ylabel='')
    st.pyplot(fig)
    st.write('Mayoritas customer tidak membutuhkan lahan parkir')
    st.markdown('---')

    # Bar Chart untuk Tipe Kamar
    st.write('#### Bar Chart of Room Type')
    fig = plt.figure(figsize=(15,5))
    data.groupby('room_type_reserved')['Booking_ID'].count().sort_values(ascending=False).plot.bar(x='room_type_reserved',
        y='booking_status', figsize=(10,10), title='What room the customer booked?')
    st.pyplot(fig)
    st.write('Customer cenderung memilih tipe room 1 untuk menginap di Liburan Skuy')
    st.markdown('---')

    # Bar Chart untuk Media Booking
    st.write('#### Bar Chart of Market Segment')
    fig = plt.figure(figsize=(15,5))
    data.groupby('market_segment_type')['Booking_ID'].count().sort_values(ascending=False).plot.bar(x='market_segment_type',
        y='booking_status', figsize=(10,10), title='Which media customer book the hotel from?')
    st.pyplot(fig)
    st.write('Customer biasanya booking hotel via Online')
    st.markdown('---')

    # Pie Chart untuk Repeated Guests
    st.write('#### Pie Chart for Repeates Guests')
    fig = plt.figure(figsize=(15,5))
    data.groupby('repeated_guest')['Booking_ID'].count().plot.pie(y='Booking_ID', autopct='%0.2f', figsize=(10,10),
        title='Is Customer a repeated guest?',ylabel='')
    st.pyplot(fig)
    st.write('Kebanyakan customer merupakan pelanggan baru')
    st.markdown('---')

    # Pie Chart untuk cek distribusi kelas
    st.write('#### Pie Chart for Booking Status')
    fig = plt.figure(figsize=(15,5))
    data.groupby('booking_status')['booking_status'].count().plot.pie(y='booking_status', autopct='%0.2f', figsize=(10,10))
    st.pyplot(fig)
    st.write('Data pada kasus ini sedikit tidak imbang atau **imbalance**. Dikatakan karena data - data dengan \
             order booking di cancel hanya sebanyak **32,76%** saja dari keseluruhan data.')
    st.markdown('---')

    # Histogram untuk cek distribusi data
    st.write('#### Histogram to see if the data skewed or not')
    total_col = len(num_col)
    fig = plt.figure(figsize=(25, total_col * 10))
    i = 1
    for col in num_col:
        plt.subplot(total_col * 4, 4, i)
        sns.histplot(data[col], kde=True, bins=30)
        plt.title(f'Histogram of {col}')
        i += 1

    plt.tight_layout()
    st.pyplot(fig)

    st.write('Dari hasil visualisasi diatas, dapat dikatakan bahwa:')
    st.write('- Data **no_of_adults**, **arrival_month** dan **arrival_date** memiliki pemusatan \
             data yang sudah terpusah ke tengah. Walaupun data **no_of_adults** didominasi dengan \
             data yang berada di tengah')
    st.write('- Data **no_of_children**, **no_of_weekends**, **no_of_week_nights**, **lead_time**, \
             **no_of_previous_cancellations**, **no_of_previous_bookings_not_canceled**, **avg_price_per_room**,\
              dan **no_of_special_requests** memiliki distribusi data yang cenderung condong ke kiri.')
    st.write('Selain itu ada beberapa hal yang bisa disimpulkan:')
    st.write('- Data diambil pada tahun **2017 dan 2018** walaupun lebih banyak order berasal dari tahun **2018**')
    st.write('- Customer yang membooking kamar biasanya terdiri dari **2 orang dewasa**')
    st.write('- Kebanyakan customer melakukan order **tidak memiliki anak**')
    st.write('Dari analisa sebelumnya kebanyakan customer merupakan customer **baru** sehingga distribusi data \
             **no_of_previous_cancellations** condong ke kiri karena customer yang melakukan order kebanyakan \
             customer baru')
    st.markdown('---')
    
    st.write('#### Boxplot to see data outlier')
    total_col = len(num_col)
    fig = plt.figure(figsize=(16, total_col * 10))
    i = 1
    for col in num_col:
        plt.subplot(total_col * 4, 4, i)
        sns.boxplot(data[col])
        plt.title(f'Boxplot of {col}')
        i += 1

    plt.tight_layout()
    st.pyplot(fig)
    st.write('Dari visualisasi boxplot, hanya data **arrival_month** dan **arrival_date** yang tidak memiliki outlier.')
    st.markdown('---')

    # Heatmap untuk korelasi data
    st.write('#### Heatmap for data correlation (Only Numerical)')
    label_encode = {'Not_Canceled':0, 'Canceled':1}
    data_label = data['booking_status'].replace(label_encode)
    data_label = pd.DataFrame(data_label,columns=['booking_status'])
    check_corr = pd.concat([data[num_col],data_label],axis=1)
    fig = plt.figure(figsize=(20,20))
    sns.heatmap(check_corr.corr(method='spearman'), annot=True)
    st.pyplot(fig)
    st.write('Hanya data **lead_time** dan **no_of_special_requests** yang memiliki korelasi atau keterkaitan dengan data\
              **booking_status** dengan nilai korelasi diatas **0.2**. Untuk data category, perlu dilakukan Chi-square test\
              untuk melakukan pengecekan keterhubungan data category dengan data **booking_status**. Dimana p-value yang\
              dibawah 0.05 memiliki keterkaitan yang signifikan dengan data **booking_status**')
    st.markdown('---')

    # Chi Test Result
    data_encode = data[cat_col].copy()
    encoder = OrdinalEncoder()
    data_encode = encoder.fit_transform(data_encode)
    chi_scores_cat = chi2(data_encode, data_label)
    p_values_cat = pd.Series(chi_scores_cat[1],index = cat_col)
    p_values_cat.sort_values(ascending = False , inplace = True)
    fig = plt.figure(figsize=(10,10))
    p_values_cat.plot.bar()
    st.pyplot(fig)
    st.write('Dari hasil analisa dan visualisasi diatas, semua data category signifikan berhubungan dengan data **booking_status**')
    st.markdown('---')
    