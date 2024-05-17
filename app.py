import os
import pickle
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
from streamlit_option_menu import option_menu

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️",
                   initial_sidebar_state="expanded")

def predict_diabetes(input_data):
    prediction = diabetes_model.predict([input_data])
    return prediction[0]

def predict_heart_disease(input_data):
    prediction = heart_disease_model.predict([input_data])
    return prediction[0]

def predict_parkinsons(input_data):
    prediction = parkinsons_model.predict([input_data])
    return prediction[0]

def load_model(file_name):
    model_path = os.path.join(os.path.dirname(__file__), 'saved_models', file_name)
    return pickle.load(open(model_path, 'rb'))

# Load models
diabetes_model = load_model('diabetes_model.sav')
heart_disease_model = load_model('heart_disease_model.sav')
parkinsons_model = load_model('parkinsons_model.sav')

# Home page
def home_page():
    st.title("Welcome to the Health Assistant")
    
    # Align and style the image properly
    st.markdown("""
    <div style="display: flex; justify-content: flex-start; margin-top: 20px;">
        <img src="https://img.freepik.com/free-vector/gradient-ai-healthcare-illustration_52683-154394.jpg?t=st=1715947643~exp=1715951243~hmac=30b43baa12fc9307d393ba493e674cd09248cc6575842eb4844c9fbc1bfcf778&w=740"
             style="width: 30%; height: auto; border: 2px solid #3498DB; border-radius: 10px; margin-right: 20px;">
    </div>
    """, unsafe_allow_html=True)
    
    # Add some padding below the image
    st.write("")
    
    # Customize overview text
    st.markdown("""
    <style>
    .overview-text {
        font-size: 20px;
        font-weight: bold;
        color: #2E4053;
        margin-top: 20px;
        padding: 20px;
        border: 2px solid #3498DB;
        border-radius: 10px;
        background-color: #E8F8F5;
        animation: fadeIn 2s;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
    
    <div class="overview-text">
    <p>Overview:</p>
    <p>This application uses Machine Learning models to predict the likelihood of an individual having one of the following diseases based on the provided inputs:</p>
    <ul>
        <li>Diabetes</li>
        <li>Heart Disease</li>
        <li>Parkinson's Disease</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Team members section only on home page
    st.markdown("""
    <style>
    .team-member {
        text-align: center;
        margin-top: 30px;
    }
    .team-member img {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        border: 2px solid #3498DB;
        margin-bottom: 10px;
    }
    .team-member .caption {
        font-weight: bold;
        color: #2E4053;
    }
    </style>
    """, unsafe_allow_html=True)

    # Add team members section
    st.markdown("<h2 style='margin-top: 40px;'>Meet Our Team</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="team-member">
            <img src="https://i.postimg.cc/kMSS0TD1/gaurav.jpg" alt="Gaurav Mishra">
            <p class="caption">Gaurav Mishra</p>
            <p>Roll No: 200333013001</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="team-member">
            <img src="https://i.postimg.cc/hjs30Kxw/zarara1.jpg" alt="Mohammad Zarara">
            <p class="caption">Mohammad Zarara</p>
            <p>Roll No: 2003330130002</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="team-member">
            <img src="https://i.postimg.cc/6pWBhpDV/utkarsh.jpg" alt="Utkarsh Trivedi">
            <p class="caption">Utkarsh Trivedi</p>
            <p>Roll No: 2003330130003</p>
        </div>
        """, unsafe_allow_html=True)
        
        
        
          # Disease Statistics Section
    st.markdown("<h2 style='margin-top: 40px;'>Disease Statistics</h2>", unsafe_allow_html=True)

    # Dummy data for demonstration
    disease_data = {
    'Disease': ['Diabetes', 'Heart Disease', 'Parkinson\'s Disease'],
    'Number of Cases': [463_000_000, 17_900_000, 10_000_000]
    }

    disease_df = pd.DataFrame(disease_data)
    fig = px.bar(disease_df, x='Disease', y='Number of Cases', title='Global Cases of Diseases', color='Disease')

    st.plotly_chart(fig)

    # Interactive Data Exploration
    st.markdown("<h2 style='margin-top: 40px;'>Explore Your Data</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=['csv'])

    if uploaded_file is not None:
        user_data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(user_data.head())

        st.write("Data Statistics:")
        st.write(user_data.describe())

        st.write("Correlation Matrix:")
        fig = px.imshow(user_data.corr(), text_auto=True, title="Correlation Matrix")
        st.plotly_chart(fig)


#  Contact Me section
    st.markdown("<h2 style='margin-top: 40px;'>Contact Me</h2>", unsafe_allow_html=True)

    # Contact form
    with st.form("contact_form", clear_on_submit=True):
        name = st.text_input("Your Name", "")
        email = st.text_input("Your Email", "")
        message = st.text_area("Your Message", "")
        submitted = st.form_submit_button("Send")
        
        if submitted:
            # You can handle the form submission here, for example, by sending an email.
            # For simplicity, we'll just display a success message.
            st.success("Message sent successfully!")





# Diabetes Prediction Page
def diabetes_page():
    st.title('Diabetes Prediction using ML')
    st.markdown("Enter the following details to predict if the person is diabetic:")

    # User inputs
    with st.form(key='diabetes_form'):
        col1, col2, col3 = st.columns(3)

        with col1:
            pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, step=1)
        with col2:
            glucose = st.number_input('Glucose Level', min_value=0, max_value=200)
        with col3:
            blood_pressure = st.number_input('Blood Pressure value', min_value=0, max_value=200)
        with col1:
            skin_thickness = st.number_input('Skin Thickness value', min_value=0, max_value=100)
        with col2:
            insulin = st.number_input('Insulin Level', min_value=0, max_value=900)
        with col3:
            bmi = st.number_input('BMI value', min_value=0.0, max_value=70.0, format="%.1f")
        with col1:
            diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function value', min_value=0.0, max_value=3.0, format="%.2f")
        with col2:
            age = st.number_input('Age of the Person', min_value=0, max_value=120)

        submit_button = st.form_submit_button(label='Diabetes Test Result')

    # Prediction
    if submit_button:
        input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
        prediction = predict_diabetes(input_data)
        result = 'The person is diabetic' if prediction == 1 else 'The person is not diabetic'
        st.success(result)

        # Visualization
        st.subheader('Input Data Overview')
        input_df = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [diabetes_pedigree_function],
            'Age': [age]
        })
        st.write(input_df)

        st.subheader('Data Distribution')
        fig = px.bar(input_df.melt(), x='variable', y='value', title='User Input Data Distribution')
        st.plotly_chart(fig)

# Heart Disease Prediction Page
def heart_disease_page():
    st.title('Heart Disease Prediction using ML')
    st.markdown("Enter the following details to predict if the person has heart disease:")

    # User inputs
    with st.form(key='heart_disease_form'):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input('Age', min_value=0, max_value=120)
        with col2:
            sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
        with col3:
            cp = st.selectbox('Chest Pain types', options=[0, 1, 2, 3])
        with col1:
            trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=200)
        with col2:
            chol = st.number_input('Serum Cholesterol in mg/dl', min_value=0, max_value=600)
        with col3:
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
        with col1:
            restecg = st.selectbox('Resting Electrocardiographic results', options=[0, 1, 2])
        with col2:
            thalach = st.number_input('Maximum Heart Rate achieved', min_value=0, max_value=250)
        with col3:
            exang = st.selectbox('Exercise Induced Angina', options=[0, 1])
        with col1:
            oldpeak = st.number_input('ST depression induced by exercise', min_value=0.0, max_value=10.0, format="%.1f")
        with col2:
            slope = st.selectbox('Slope of the peak exercise ST segment', options=[0, 1, 2])
        with col3:
            ca = st.number_input('Major vessels colored by fluoroscopy', min_value=0, max_value=4)
        with col1:
            thal = st.selectbox('Thalassemia', options=[0, 1, 2], format_func=lambda x: ['Normal', 'Fixed defect', 'Reversible defect'][x])

        submit_button = st.form_submit_button(label='Heart Disease Test Result')

    # Prediction
    if submit_button:
        input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        prediction = predict_heart_disease(input_data)
        result = 'The person has heart disease' if prediction == 1 else 'The person does not have heart disease'
        st.success(result)

        # Visualization
        st.subheader('Input Data Overview')
        input_df = pd.DataFrame({
            'Age': [age],
            'Sex': [sex],
            'Chest Pain': [cp],
            'Resting BP': [trestbps],
            'Cholesterol': [chol],
            'Fasting BS': [fbs],
            'Rest ECG': [restecg],
            'Max HR': [thalach],
            'Exercise Induced Angina': [exang],
            'Old Peak': [oldpeak],
            'Slope': [slope],
            'CA': [ca],
            'Thal': [thal]
        })
        st.write(input_df)

        st.subheader('Data Distribution')
        fig = px.bar(input_df.melt(), x='variable', y='value', title='User Input Data Distribution')
        st.plotly_chart(fig)

# Parkinson's Prediction Page
def parkinsons_page():
    st.title("Parkinson's Disease Prediction using ML")
    st.markdown("Enter the following details to predict if the person has Parkinson's disease:")

    # User inputs
    with st.form(key='parkinsons_form'):
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            fo = st.number_input('MDVP:Fo(Hz)', min_value=0.0, max_value=300.0, format="%.2f")
        with col2:
            fhi = st.number_input('MDVP:Fhi(Hz)', min_value=0.0, max_value=300.0, format="%.2f")
        with col3:
            flo = st.number_input('MDVP:Flo(Hz)', min_value=0.0, max_value=300.0, format="%.2f")
        with col4:
            jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0, max_value=1.0, format="%.4f")
        with col5:
            jitter_abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0, max_value=0.1, format="%.6f")
        with col1:
            rap = st.number_input('MDVP:RAP', min_value=0.0, max_value=0.1, format="%.4f")
        with col2:
            ppq = st.number_input('MDVP:PPQ', min_value=0.0, max_value=0.1, format="%.4f")
        with col3:
            ddp = st.number_input('Jitter:DDP', min_value=0.0, max_value=0.1, format="%.4f")
        with col4:
            shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, max_value=1.0, format="%.4f")
        with col5:
            shimmer_db = st.number_input('MDVP:Shimmer(dB)', min_value=0.0, max_value=10.0, format="%.2f")
        with col1:
            apq3 = st.number_input('Shimmer:APQ3', min_value=0.0, max_value=1.0, format="%.4f")
        with col2:
            apq5 = st.number_input('Shimmer:APQ5', min_value=0.0, max_value=1.0, format="%.4f")
        with col3:
            apq = st.number_input('MDVP:APQ', min_value=0.0, max_value=1.0, format="%.4f")
        with col4:
            dda = st.number_input('Shimmer:DDA', min_value=0.0, max_value=1.0, format="%.4f")
        with col5:
            nhr = st.number_input('NHR', min_value=0.0, max_value=1.0, format="%.4f")
        with col1:
            hnr = st.number_input('HNR', min_value=0.0, max_value=100.0, format="%.2f")
        with col2:
            rpde = st.number_input('RPDE', min_value=0.0, max_value=1.0, format="%.4f")
        with col3:
            dfa = st.number_input('DFA', min_value=0.0, max_value=3.0, format="%.4f")
        with col4:
            spread1 = st.number_input('spread1', min_value=-7.0, max_value=0.0, format="%.4f")
        with col5:
            spread2 = st.number_input('spread2', min_value=0.0, max_value=7.0, format="%.4f")
        with col1:
            d2 = st.number_input('D2', min_value=0.0, max_value=7.0, format="%.4f")
        with col2:
            ppe = st.number_input('PPE', min_value=0.0, max_value=1.0, format="%.4f")

        submit_button = st.form_submit_button(label="Parkinson's Test Result")

    # Prediction
    if submit_button:
        input_data = [fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]
        prediction = predict_parkinsons(input_data)
        result = "The person has Parkinson's disease" if prediction == 1 else "The person does not have Parkinson's disease"
        st.success(result)

        # Visualization
        st.subheader('Input Data Overview')
        input_df = pd.DataFrame({
            'MDVP:Fo(Hz)': [fo],
            'MDVP:Fhi(Hz)': [fhi],
            'MDVP:Flo(Hz)': [flo],
            'MDVP:Jitter(%)': [jitter_percent],
            'MDVP:Jitter(Abs)': [jitter_abs],
            'MDVP:RAP': [rap],
            'MDVP:PPQ': [ppq],
            'Jitter:DDP': [ddp],
            'MDVP:Shimmer': [shimmer],
            'MDVP:Shimmer(dB)': [shimmer_db],
            'Shimmer:APQ3': [apq3],
            'Shimmer:APQ5': [apq5],
            'MDVP:APQ': [apq],
            'Shimmer:DDA': [dda],
            'NHR': [nhr],
            'HNR': [hnr],
            'RPDE': [rpde],
            'DFA': [dfa],
            'spread1': [spread1],
            'spread2': [spread2],
            'D2': [d2],
            'PPE': [ppe]
        })
        st.write(input_df)

        st.subheader('Data Distribution')
        fig = px.bar(input_df.melt(), x='variable', y='value', title='User Input Data Distribution')
        st.plotly_chart(fig)

# Sidebar navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Home', 'Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
                           menu_icon='hospital-fill',
                           icons=['house', 'activity', 'heart', 'person'],
                           default_index=0)

# Page rendering based on selection
if selected == 'Home':
    home_page()
elif selected == 'Diabetes Prediction':
    diabetes_page()
elif selected == 'Heart Disease Prediction':
    heart_disease_page()
elif selected == "Parkinsons Prediction":
    parkinsons_page()
