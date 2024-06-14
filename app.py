import pickle
import streamlit as st
import pandas as pd
import numpy as np


@st.cache(suppress_st_warning=True)
def get_fvalue(val):
    feature_dict = {"No": 1, "Yes": 2}
    for key, value in feature_dict.items():
        if val == key:
            return value


def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value


app_mode = st.sidebar.selectbox(
    'Select Page', ['Home', 'Prediction'])  # two pages

if app_mode == "Home":
    st.title('Diabetes')
    st.caption("Database diambil dari Institusi Nasional Diabetes dan Penyakit Pencenaan dan Ginjal")
    st.caption('''Database terdiri 768 orang, 500 orang tidak terdeteksi penyakit diabetes dan 268 orang terdeteksi menderita penyakit diabetes. 
               Pada data diabetes ini terdiri dari 9 atribut, 8 atribut predictor dan 1 atribut tujuan.''')
    st.markdown('Dataset :')
    dataset = pd.read_csv('diabetes.csv')
    items_per_page = 20
    page_number = st.number_input('Enter page number:', min_value=1, value=1)
    start_idx = (page_number - 1) * items_per_page
    end_idx = start_idx + items_per_page
    st.table(dataset.iloc[start_idx:end_idx])
    
elif app_mode == "Prediction":
    st.title('Prediksi Diabetes')
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input('Input Nilai Pregnancies (Berapa Kali Hamil)', min_value=0, max_value=17, value=0)
        Glucose = st.number_input('Input Nilai Glucose (mg/dL)', min_value=0, max_value=200, value=0)
        BloodPressure = st.number_input('Input Nilai BloodPressure (mmHg)', min_value=0, max_value=250, value=0)

    with col2:
        SkinThickness = st.number_input('Input Nilai SkinThickness/Ketebalan Kulit (milimeter)', min_value=0, max_value=200, value=0)
        Insulin = st.number_input('Input Nilai Insulin', min_value=0, max_value=850, value=0)
        BMI = st.number_input('Input Nilai  BMI (Kg/m^2)', min_value=0.00, max_value=100.00, value=0.00)

    with col3:
        DiabetesPedigreeFunction = st.number_input('Input Nilai  DiabetesPedigreeFunction (Riwat)', min_value=0.00, max_value=3.00, value=0.00)
        Age = st.number_input('Input Nilai  Age (Tahun)', min_value=0, max_value=80, value=0)

    feature_list = [Pregnancies, Glucose, BloodPressure,
                    SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    single_sampel = np.array(feature_list).reshape(1, -1)
    if st.button('Test Prediksi Diabetes'):
        file_model = pickle.load(open('diabetes_model.sav', 'rb'))
        prediction = file_model.predict(single_sampel)
        if prediction[0] == 1:
            st.error('Terindeksi Penyakit Diabetes Dengan Accuracy 79.2%')
        elif prediction[0] == 0:
            st.success('Tidak Terindeksi  Penyakit Diabetes Dengan Accuracy 79.2%')
            
    st.caption('Keterangan:') 
    st.caption(' 1. Pregnancies = Jumlah Kehamilan Yang Sudah Dialami')
    st.caption(' 2. Glucose = Jumlah Kadar Gula Dalam Tubuh')
    st.caption(' 3. Blood Pressure = Tekanan Darah Pada Tubuh (Tekanan darah yang digunakan = Tekanan Sistolik)')
    st.caption(' 4. SkinThickness = Ketebalan Kulit (Ukuran Fisik dari Lapisan Kulit)')
    st.caption(' 5. Insulin = Jumlah Insulin dalam Darah')
    st.caption(' 6. BMI = Hasil Pembagian Berat Badan dengan Tinggi Badan Pangkat Dua')
    st.caption(''' 7. Diabetes Pedigree Function = Jumlah semua kelurga punya penyakit diabetes 
               (0.01 * usia kelurga yang punya penyakit diabetes) 
               ditambah jumlah hubungan dekat dengan pasien yang diuji. 
               Nilai untuk hubungan dekat : 0.0 untuk bukan orang tua atau saudara kandung,
               0.5 untuk orang tua atau saudara kandung, dan 1.0 untuk orang tua dan saudara kandung''')
    st.caption(' 8. Age = Umur Seseorang')