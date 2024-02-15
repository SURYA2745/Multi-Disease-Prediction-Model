import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
from PIL import Image
import numpy as np
import plotly.figure_factory as ff
import streamlit as st
from train.DiseaseModel import DiseaseModel
from train.helper import prepare_symptoms_array
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

import pickle
import sqlite3
from pathlib import Path
import streamlit_authenticator as stauth
#from auth import Authenticate

diabetes_model = joblib.load("models/diabetes_model.sav")
heart_model = joblib.load("models/heart_disease_model.sav")
lung_cancer_model = joblib.load('models/lung_cancer_model.sav')
chronic_disease_model = joblib.load('models/chronic_model.sav')
hepatitis_model = joblib.load('models/hepititisc_model.sav')

st.markdown("<h1 style='text-align: center; color: white;'>🏥Multi Disease Prediction Model</h1>", unsafe_allow_html=True)

import yaml
from yaml.loader import SafeLoader
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)
authenticator.login()
if st.session_state["authentication_status"]:
    
    st.sidebar.title(f'Welcome, *{st.session_state["name"]}*😃')
    #st.title('Some content')
    # sidebar
    with st.sidebar:
        selected = option_menu('Our Services🩺', [
            'Dashboard',
            'View Train & Test Data',
            'Disease Prediction',
            'Diabetes Prediction',
            'Heart Disease Prediction',
            'Hepatitis Prediction',
            'Lung Cancer Prediction',
            'Chronic Kidney Prediction',
            'Classification Reports',
            'Contact us'
        ],
            icons=['bar-chart-fill','person', 'activity', 'heart','person','lungs','person','bar-chart-fill','','person'],
            default_index=0)
        authenticator.logout()
    # Dashboard page
    if selected == 'Dashboard': 
        image = Image.open('images/Dash.jpeg')
        st.image(image, caption='')
        #st.markdown("<h1 style='text-align: center; color: black;'>Multi Disease Prediction Model</h1>", unsafe_allow_html=True)
        st.title('Disease Prediction using Machine Learning 📈')
        st.write('Disease Prediction using Machine Learning is the system that is used to predict the diseases from the symptoms which are given by the patients or any user. The system processes the symptoms provided by the user as input and gives the output as the probability of the disease. XGBoost is used in the prediction of the disease which is a supervised machine learning algorithm. The probability of the disease is calculated by the XGBoost Algorithm. With an increase in biomedical and healthcare data, accurate analysis of medical data benefits early disease detection and patient care. By using Linear regression, Random forest, BaggingClassifier, AdaBoostClassifier and SVM. we are predicting diseases like Diabetes, Heart, Hepatitis, Lung cancer and Chronic Kidnet diseases.')
        
        st.title('Diabetes Disease 🧬')
        st.write('Diabetes is a condition that happens when your blood sugar (glucose) is too high. It develops when your pancreas doesn’t make enough insulin or any at all, or when your body isn’t responding to the effects of insulin properly. Diabetes affects people of all ages. Most forms of diabetes are chronic (lifelong), and all forms are manageable with medications and/or lifestyle changes.')
        st.write('Glucose (sugar) mainly comes from carbohydrates in your food and drinks. It’s your body’s go-to source of energy. Your blood carries glucose to all your body’s cells to use for energy.')
        image = Image.open('images/diabetes.png')
        st.image(image, caption='Diabetes Complications')
        
        st.title('Heart Disease 🫀')
        st.write('Heart disease is a variety of issues that can affect your heart. When people think about heart disease, they often think of the most common type — coronary artery disease (CAD) and the heart attacks it can cause. But you can have trouble with different parts of your heart, like your heart muscle, valves or electrical system and other problems etc.')
        st.write('When your heart isn’t working well, it has trouble sending enough blood, oxygen and nutrients to your body. In a way, your heart delivers the fuel that keeps your body’s systems running. If there’s a problem with delivering that fuel, it affects everything your body’s systems do. Healthy habits, medicines and procedures can prevent or treat CAD and other heart diseases.Lifestyle changes and medications can keep your heart healthy and lower your chances of getting heart disease.')
        image = Image.open('images/heart2.jpg')
        st.image(image, caption='Heart Complications')
        
        st.title('Hepatits Disease 🦠')
        st.write('Hepatitis is inflammation of the liver. Inflammation is swelling that happens when tissues of the body are injured or infected. It can damage your liver. This swelling and damage can affect how well your liver functions.Hepatitis can be an acute (short-term) infection or a chronic (long-term) infection. Some types of hepatitis cause only acute infections. Other types can cause both acute and chronic infections.')
        st.write('There are several types of hepatitis, which are categorized as viral hepatitis and non-viral hepatitis.The most common types of viral hepatitis are: A, B, C, D, and E.Then there are non-viral causes of hepatitis, such as autoimmune hepatitis, alcoholic hepatitis, and drug-induced hepatitis. These forms of hepatitis are not caused by viral infections but rather by non-infectious causes such as autoimmune disorders, excessive alcohol consumption, or certain medications or toxins.')
        image = Image.open('images/hepatitis1.jpg')
        st.image(image, caption='Hepatitis C Complications')
        
        st.title('Lung Cancer Disease 🫁')
        st.write('Lung cancer is a type of cancer that begins in the lungs. Your lungs are two spongy organs in your chest that take in oxygen when you inhale and release carbon dioxide when you exhale.Lung cancer is the leading cause of cancer deaths worldwide.')
        st.write('People who smoke have the greatest risk of lung cancer, though lung cancer can also occur in people who have never smoked. The risk of lung cancer increases with the length of time and number of cigarettes you have smoked. If you quits smoking, even after smoking many years, you can significantly reduce your chances of developing lung cancer.')
        image = Image.open('images/lungs.jpg')
        st.image(image, caption='Lung Cancer Stages')
        
        st.title('Chronic Kidney Disease 💨')
        st.write('Chronic kidney disease, also called chronic kidney failure, involves a gradual loss of kidney function. Your kidneys filter wastes and excess fluids from your blood, which are then removed in your urine. Advanced chronic kidney disease can cause dangerous levels of fluid, electrolytes and wastes to build up in your body.')
        st.write('In the early stages of chronic kidney disease, you might have few signs or symptoms. You might not realize that you have kidney disease until the condition is advanced.')
        st.write('Treatment for chronic kidney disease focuses on slowing the progression of kidney damage, usually by controlling the cause. But, even controlling the cause might not keep kidney damage from progressing. Chronic kidney disease can progress to end-stage kidney failure, which is fatal without artificial filtering (dialysis) or a kidney transplant.')
        image = Image.open('images/kidney.jpg')
        st.image(image, caption='Kidney Failure Stages')
    if selected == 'View Train & Test Data':
        with st.container():
            selected1 = option_menu(
                menu_title=None,
                options=['Training Dataset','Testing Dataset'],
                icons=['activity','code-slash'],
                orientation='horizontal'
            )
            if selected1=='Training Dataset':
                df = pd.read_csv("data/Training.csv")
                st.title('Training Data')
                st.write(df)
            if selected1=='Testing Dataset':
                df = pd.read_csv("data/Testing.csv")
                st.title('Testing Data')
                st.write(df)
            
    # Connect to SQLite and create a table for multiple diseases predictions
    def create_disease_table():
        connection = None
        try:
            connection = sqlite3.connect('disease_predictions.db')
            cursor = connection.cursor()

            # Define the table schema
            create_table_query = '''
                CREATE TABLE IF NOT EXISTS disease_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symptoms TEXT NOT NULL,
                    predicted_disease TEXT,
                    precaution1 TEXT,
                    precaution2 TEXT,
                    precaution3 TEXT,
                    precaution4 TEXT
                )
            '''

            # Execute the query to create the table
            cursor.execute(create_table_query)
            connection.commit()

            print("Table 'disease_predictions' created successfully in SQLite database")

        except sqlite3.Error as e:
            print(f"Error: {e}")

        finally:
            if connection:
                connection.close()
                print("SQLite connection closed")

    # Function to insert data into SQLite for multiple diseases
    def insert_disease_data_sqlite(symptoms, predicted_disease, prediction_probability, precautions):
        connection = sqlite3.connect('disease_predictions.db')
        cursor = connection.cursor()

        try:
            sql_query = "INSERT INTO disease_predictions (symptoms, predicted_disease, precaution1, precaution2, precaution3, precaution4) " \
                        "VALUES (?, ?, ?, ?, ?, ?)"
            values = (symptoms, predicted_disease, precautions[0], precautions[1], precautions[2], precautions[3])

            cursor.execute(sql_query, values)
            connection.commit()
            print("Data inserted into disease_predictions table in SQLite successfully")

        except sqlite3.Error as e:
            print(f"Error: {e}")

        finally:
            if connection:
                cursor.close()
                connection.close()
                print("SQLite connection closed")

    # Call the function to create the table for disease predictions
    create_disease_table()
    # multiple disease prediction
    if selected == 'Disease Prediction': 
        # Create disease class and load ML model
        disease_model = DiseaseModel()
        disease_model.load_xgboost('model/xgboost_model.json')

        # Title
        st.write('# Check Your Disease')
        image = Image.open('images/disease.jpeg')
        st.image(image, caption='Check your Disease')
        symptoms = st.multiselect('Enter your symptoms?', options=disease_model.all_symptoms)

        X = prepare_symptoms_array(symptoms)

        # Trigger XGBoost model
        if st.button('Predict the result'): 
            if not symptoms:
                st.warning("Please enter at least Two or Three symptoms.")
            # Run the model with the python script
            
            prediction, prob = disease_model.predict(X)
            st.write(f'## Disease: {prediction} with {prob*100:.2f}% probability')


            tab1, tab2= st.tabs(["Description", "Precautions"])

            with tab1:
                st.write(disease_model.describe_predicted_disease())

            with tab2:
                precautions = disease_model.predicted_disease_precautions()
                for i in range(4):
                    st.write(f'{i+1}. {precautions[i]}')
                    
            # Insert data into SQLite database
            insert_disease_data_sqlite(','.join(symptoms), prediction,prob,precautions)
    # Connect to SQLite and create a table
    def create_table():
        connection = None
        try:
            connection = sqlite3.connect('diabetes_predictions.db')
            cursor = connection.cursor()

            # Define the table schema
            create_table_query = '''
                CREATE TABLE IF NOT EXISTS diabetes_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    pregnancies INTEGER,
                    glucose INTEGER,
                    blood_pressure INTEGER,
                    skin_thickness INTEGER,
                    insulin INTEGER,
                    bmi REAL,
                    pedigree_function REAL,
                    age INTEGER,
                    prediction_result TEXT
                )
            '''

            # Execute the query to create the table
            cursor.execute(create_table_query)
            connection.commit()

            print("Table 'diabetes_predictions' created successfully in SQLite database")

        except sqlite3.Error as e:
            print(f"Error: {e}")

        finally:
            if connection:
                connection.close()
                print("SQLite connection closed")

    # Function to insert data into SQLite
    def insert_data_sqlite(name, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree_function, age, prediction_result):
            connection = sqlite3.connect('diabetes_predictions.db')
            cursor = connection.cursor()

            try:
                sql_query = "INSERT INTO diabetes_predictions (name, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree_function, age, prediction_result) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                values = (name, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree_function, age, prediction_result)

                cursor.execute(sql_query, values)
                connection.commit()
                print("Data inserted into SQLite successfully")

            except sqlite3.Error as e:
                print(f"Error: {e}")

            finally:
                if connection:
                    cursor.close()
                    connection.close()
                    print("SQLite connection closed")

        # Call the function to create the table
    create_table()

    # Diabetes prediction page
    if selected == 'Diabetes Prediction':  # pagetitle
        st.title("Diabetes Disease Prediction")
        image = Image.open('images/diabetes1.jpg')
        st.image(image, caption='Diabetes Disease Prediction')
        st.subheader('Diabetes Symptom Ranges 📊')
        st.write('Glucose Level: 100-125 or Higher mg/dL')
        st.write('Blood Pressure Level: < 140/90 mm Hg')
        st.write('Skin Thickness value: 1.9 to 2.4 mm')
        st.write('Insulin Value: 5 to 15 U/ML')
        st.write('BMI Values: 30 to 39.9 kg/m2')
        st.write('Diabetes PedigreeFunction: 0.08 to 2.42')
        st.write('Note: The above values are taken for only prediction purpose not exact values !')
        # columns
        # no inputs from the user
        name = st.text_input("Name:")
        col1, col2, col3 = st.columns(3)

        with col1:
            Pregnancies = st.number_input("Number of Pregnencies")
        with col2:
            Glucose = st.number_input("Glucose Level")
        with col3:
            BloodPressure = st.number_input("Blood Pressure  Value")
        with col1:

            SkinThickness = st.number_input("Skin Thickness Value")

        with col2:

            Insulin = st.number_input("Insulin Value ")
        with col3:
            BMI = st.number_input("BMI Value")
        with col1:
            DiabetesPedigreefunction = st.number_input(
                "Diabetes Pedigreefunction Value")
        with col2:

            Age = st.number_input("AGE")

        # code for prediction
        diabetes_dig = ''

        # button
        if st.button("Diabetes test result"):
            diabetes_prediction=[[]]
            diabetes_prediction = diabetes_model.predict(
                [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreefunction, Age]])

            # after the prediction is done if the value in the list at index is 0 is 1 then the person is diabetic
            if diabetes_prediction[0] == 1:
                diabetes_dig = "we are really sorry to say, The model predicts that it seems like you are Diabetic."
                insert_data_sqlite(name, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreefunction, Age, "Diabetic")
                image = Image.open('positive.png')
                st.image(image, caption='')
                
            else:
                diabetes_dig = 'Congratulation,The model predicts that You are not diabetic'
                insert_data_sqlite(name, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreefunction, Age, "Not Diabetic")
                image = Image.open('negative.png')
                st.image(image, caption='')
            st.success(name+' , ' + diabetes_dig)
    # Connect to SQLite and create a table
    def create_heart_table():
        connection = None
        try:
            connection = sqlite3.connect('heart_predictions.db')
            cursor = connection.cursor()

            # Define the table schema
            create_table_query = '''
                CREATE TABLE IF NOT EXISTS heart_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    age INTEGER,
                    sex INTEGER,
                    cp INTEGER,
                    trestbps INTEGER,
                    chol INTEGER,
                    fbs INTEGER,
                    restecg INTEGER,
                    thalach INTEGER,
                    exang INTEGER,
                    oldpeak REAL,
                    slope INTEGER,
                    ca INTEGER,
                    thal INTEGER,
                    prediction_result TEXT
                )
            '''

            # Execute the query to create the table
            cursor.execute(create_table_query)
            connection.commit()

            print("Table 'heart_predictions' created successfully in SQLite database")

        except sqlite3.Error as e:
            print(f"Error: {e}")

        finally:
            if connection:
                connection.close()
                print("SQLite connection closed")

    # Function to insert data into SQLite
    def insert_heart_data_sqlite(name, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, prediction_result):
        connection = sqlite3.connect('heart_predictions.db')
        cursor = connection.cursor()

        try:
            sql_query = "INSERT INTO heart_predictions (name, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, prediction_result) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            values = (name, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, prediction_result)

            cursor.execute(sql_query, values)
            connection.commit()
            print("Data inserted into heart_predictions table in SQLite successfully")

        except sqlite3.Error as e:
            print(f"Error: {e}")

        finally:
            if connection:
                cursor.close()
                connection.close()
                print("SQLite connection closed")

    # Call the function to create the table for heart disease predictions
    create_heart_table()
            
    # Heart prediction page
    if selected == 'Heart Disease Prediction':
        st.title("Heart disease prediction")
        image = Image.open('images/heart.png')
        st.image(image, caption='heart failuire')
        st.subheader('Heart Disease Symptom Ranges 📊')
        st.write('Resting Blood Pressure: 120/80 mmHg')
        st.write('Serum Cholestrol: 100 to 129 mg/dL')
        st.write('Max Heart Rate Achieved: 80 to 90 bpm')
        st.write('ST depression Levels: 1 to 1.5 mm')
        st.write('major vessels colored by flourosopy: 0 to 3')
        st.write('Note: The above values are taken for only prediction purpose not exact values !')
        # age	sex	cp	trestbps	chol	fbs	restecg	thalach	exang	oldpeak	slope	ca	thal	target
        # columns
        # no inputs from the user
        name = st.text_input("Name:")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age")
        with col2:
            sex=0
            display = ("male", "female")
            options = list(range(len(display)))
            value = st.selectbox("Gender", options, format_func=lambda x: display[x])
            if value == "male":
                sex = 1
            elif value == "female":
                sex = 0
        with col3:
            cp=0
            display = ("typical angina","atypical angina","non — anginal pain","asymptotic")
            options = list(range(len(display)))
            value = st.selectbox("Chest_Pain Type", options, format_func=lambda x: display[x])
            if value == "typical angina":
                cp = 0
            elif value == "atypical angina":
                cp = 1
            elif value == "non — anginal pain":
                cp = 2
            elif value == "asymptotic":
                cp = 3
        with col1:
            trestbps = st.number_input("Resting Blood Pressure")

        with col2:

            chol = st.number_input("Serum Cholestrol")
        
        with col3:
            restecg=0
            display = ("normal","having ST-T wave abnormality","left ventricular hyperthrophy")
            options = list(range(len(display)))
            value = st.selectbox("Resting ECG", options, format_func=lambda x: display[x])
            if value == "normal":
                restecg = 0
            elif value == "having ST-T wave abnormality":
                restecg = 1
            elif value == "left ventricular hyperthrophy":
                restecg = 2

        with col1:
            exang=0
            thalach = st.number_input("Max Heart Rate Achieved")
    
        with col2:
            oldpeak = st.number_input("ST depression induced by exercise relative to rest")
        with col3:
            slope=0
            display = ("upsloping","flat","downsloping")
            options = list(range(len(display)))
            value = st.selectbox("Peak exercise ST segment", options, format_func=lambda x: display[x])
            if value == "upsloping":
                slope = 0
            elif value == "flat":
                slope = 1
            elif value == "downsloping":
                slope = 2
        with col1:
            ca = st.number_input("Number of major vessels (0–3) colored by flourosopy")
        with col2:
            thal=0
            display = ("normal","fixed defect","reversible defect")
            options = list(range(len(display)))
            value = st.selectbox("thalassemia", options, format_func=lambda x: display[x])
            if value == "normal":
                thal = 0
            elif value == "fixed defect":
                thal = 1
            elif value == "reversible defect":
                thal = 2
        with col3:
            agree = st.checkbox('Exercise induced angina')
            if agree:
                exang = 1
            else:
                exang=0
        with col1:
            agree1 = st.checkbox('fasting blood sugar > 120mg/dl')
            if agree1:
                fbs = 1
            else:
                fbs=0
        # code for prediction
        heart_dig = ''
        

        # button
        if st.button("Heart test result"):
            heart_prediction=[[]]
            # change the parameters according to the model
            
            # b=np.array(a, dtype=float)
            heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

            if heart_prediction[0] == 1:
                heart_dig = 'we are really sorry to say, The model predicts that it seems like you have Heart Disease.'
                insert_heart_data_sqlite(name, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, "+ve for Heart Disease")
                image = Image.open('positive.png')
                st.image(image, caption='')
                
            else:
                heart_dig = "Congratulation,The model predicts that You don't have Heart Disease."
                insert_heart_data_sqlite(name, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, "No Heart Disease")
                image = Image.open('negative.png')
                st.image(image, caption='')
            st.success(name +' , ' + heart_dig)
            
    # Lung Cancer Prediction page
    # Load the dataset
    lung_cancer_data = pd.read_csv('data/lung_cancer.csv')
    # Convert 'M' to 0 and 'F' to 1 in the 'GENDER' column
    lung_cancer_data['GENDER'] = lung_cancer_data['GENDER'].map({'M': 'Male', 'F': 'Female'})
    # Connect to SQLite and create a table
    def create_lung_cancer_table():
        connection = None
        try:
            connection = sqlite3.connect('lung_cancer_predictions.db')
            cursor = connection.cursor()

            # Define the table schema
            create_table_query = '''
                CREATE TABLE IF NOT EXISTS lung_cancer_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    gender TEXT,
                    age INTEGER,
                    smoking TEXT,
                    yellow_fingers TEXT,
                    anxiety TEXT,
                    peer_pressure TEXT,
                    chronic_disease TEXT,
                    fatigue INTEGER,
                    allergy INTEGER,
                    wheezing INTEGER,
                    alcohol_consuming TEXT,
                    coughing TEXT,
                    shortness_of_breath TEXT,
                    swallowing_difficulty TEXT,
                    chest_pain TEXT,
                    prediction_result TEXT
                )
            '''

            # Execute the query to create the table
            cursor.execute(create_table_query)
            connection.commit()

            print("Table 'lung_cancer_predictions' created successfully in SQLite database")

        except sqlite3.Error as e:
            print(f"Error: {e}")

        finally:
            if connection:
                connection.close()
                print("SQLite connection closed")

    # Function to insert data into SQLite
    def insert_lung_cancer_data_sqlite(name, gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease,
                                    fatigue, allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath,
                                    swallowing_difficulty, chest_pain, prediction_result):
        connection = sqlite3.connect('lung_cancer_predictions.db')
        cursor = connection.cursor()

        try:
            sql_query = "INSERT INTO lung_cancer_predictions (name, gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, " \
                        "fatigue, allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath, swallowing_difficulty, chest_pain, prediction_result) " \
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            values = (name, gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease,
                    fatigue, allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath,
                    swallowing_difficulty, chest_pain, prediction_result)

            cursor.execute(sql_query, values)
            connection.commit()
            print("Data inserted into lung_cancer_predictions table in SQLite successfully")

        except sqlite3.Error as e:
            print(f"Error: {e}")

        finally:
            if connection:
                cursor.close()
                connection.close()
                print("SQLite connection closed")

    # Call the function to create the table for lung cancer predictions
    create_lung_cancer_table()
    # Lung Cancer prediction page
    if selected == 'Lung Cancer Prediction':
        st.title("Lung Cancer Prediction")
        image = Image.open('images/lung.jpeg')
        st.image(image, caption='Lung Cancer Prediction')
        st.subheader('Lung Cancer Symptoms 📊')
        st.write('Smoking: Yes ✅ or No ❌')
        st.write('Yellow Fingers: Yes ✅ or No ❌')
        st.write('Anxiety: Yes ✅ or No ❌')
        st.write('Chronic Disease: Yes ✅ or No ❌')
        st.write('Peer Pressure: Yes ✅ or No ❌')
        st.write('Allergy: Yes ✅ or No ❌')
        st.write('Coughing: Yes ✅ or No ❌')
        st.write('Alocohol: Yes ✅ or No ❌')
        # Columns
        # No inputs from the user
        name = st.text_input("Name:")
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender:", lung_cancer_data['GENDER'].unique())
        with col2:
            age = st.number_input("Age")
        with col3:
            smoking = st.selectbox("Smoking:", ['NO', 'YES'])
        with col1:
            yellow_fingers = st.selectbox("Yellow Fingers:", ['NO', 'YES'])

        with col2:
            anxiety = st.selectbox("Anxiety:", ['NO', 'YES'])
        with col3:
            peer_pressure = st.selectbox("Peer Pressure:", ['NO', 'YES'])
        with col1:
            chronic_disease = st.selectbox("Chronic Disease:", ['NO', 'YES'])

        with col2:
            fatigue = st.selectbox("Fatigue:", ['NO', 'YES'])
        with col3:
            allergy = st.selectbox("Allergy:", ['NO', 'YES'])
        with col1:
            wheezing = st.selectbox("Wheezing:", ['NO', 'YES'])

        with col2:
            alcohol_consuming = st.selectbox("Alcohol Consuming:", ['NO', 'YES'])
        with col3:
            coughing = st.selectbox("Coughing:", ['NO', 'YES'])
        with col1:
            shortness_of_breath = st.selectbox("Shortness of Breath:", ['NO', 'YES'])

        with col2:
            swallowing_difficulty = st.selectbox("Swallowing Difficulty:", ['NO', 'YES'])
        with col3:
            chest_pain = st.selectbox("Chest Pain:", ['NO', 'YES'])

        # Code for prediction
        cancer_result = ''

        # Button
        if st.button("Predict Lung Cancer"):
            # Create a DataFrame with user inputs
            user_data = pd.DataFrame({
                'GENDER': [gender],
                'AGE': [age],
                'SMOKING': [smoking],
                'YELLOW_FINGERS': [yellow_fingers],
                'ANXIETY': [anxiety],
                'PEER_PRESSURE': [peer_pressure],
                'CHRONICDISEASE': [chronic_disease],
                'FATIGUE': [fatigue],
                'ALLERGY': [allergy],
                'WHEEZING': [wheezing],
                'ALCOHOLCONSUMING': [alcohol_consuming],
                'COUGHING': [coughing],
                'SHORTNESSOFBREATH': [shortness_of_breath],
                'SWALLOWINGDIFFICULTY': [swallowing_difficulty],
                'CHESTPAIN': [chest_pain]
            })

            # Map string values to numeric
            user_data.replace({'NO': 1, 'YES': 2}, inplace=True)

            # Strip leading and trailing whitespaces from column names
            user_data.columns = user_data.columns.str.strip()

            # Convert columns to numeric where necessary
            numeric_columns = ['AGE', 'FATIGUE', 'ALLERGY', 'ALCOHOLCONSUMING', 'COUGHING', 'SHORTNESSOFBREATH']
            user_data[numeric_columns] = user_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

            # Perform prediction
            cancer_prediction = lung_cancer_model.predict(user_data)

            # Display result
            if cancer_prediction[0] == 'YES':
                cancer_result = "We are really sorry to say, The model predicts that there is a risk of Lung Cancer."
                insert_lung_cancer_data_sqlite(name, gender, age, smoking, yellow_fingers, anxiety, peer_pressure,chronic_disease, fatigue, allergy, wheezing, alcohol_consuming, coughing,shortness_of_breath, swallowing_difficulty, chest_pain, "Risk of Lung Cancer")
                image = Image.open('positive.png')
                st.image(image, caption='')
            else:
                cancer_result = "Congratulations, The model predicts no significant risk of Lung Cancer."
                insert_lung_cancer_data_sqlite(name, gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol_consuming, coughing,shortness_of_breath, swallowing_difficulty, chest_pain, "No Risk of Lung Cancer")
                image = Image.open('negative.png')
                st.image(image, caption='')

            st.success(name + ', ' + cancer_result)
    # Connect to SQLite and create a table
    def create_hepatitis_table():
        connection = None
        try:
            connection = sqlite3.connect('hepatitis_predictions.db')
            cursor = connection.cursor()

            # Define the table schema
            create_table_query = '''
                CREATE TABLE IF NOT EXISTS hepatitis_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    age INTEGER,
                    sex INTEGER,
                    total_bilirubin REAL,
                    direct_bilirubin REAL,
                    alkaline_phosphatase REAL,
                    alamine_aminotransferase REAL,
                    aspartate_aminotransferase REAL,
                    total_proteins REAL,
                    albumin REAL,
                    albumin_and_globulin_ratio REAL,
                    ggt_value REAL,
                    prot_value REAL,
                    prediction_result TEXT
                )
            '''

            # Execute the query to create the table
            cursor.execute(create_table_query)
            connection.commit()

            print("Table 'hepatitis_predictions' created successfully in SQLite database")

        except sqlite3.Error as e:
            print(f"Error: {e}")

        finally:
            if connection:
                connection.close()
                print("SQLite connection closed")

    # Function to insert data into SQLite
    def insert_hepatitis_data_sqlite(name, age, sex, total_bilirubin, direct_bilirubin, alkaline_phosphatase, alamine_aminotransferase, aspartate_aminotransferase, total_proteins, albumin, albumin_and_globulin_ratio, ggt_value, prot_value, prediction_result):
        connection = sqlite3.connect('hepatitis_predictions.db')
        cursor = connection.cursor()

        try:
            sql_query = "INSERT INTO hepatitis_predictions (name, age, sex, total_bilirubin, direct_bilirubin, alkaline_phosphatase, alamine_aminotransferase, aspartate_aminotransferase, total_proteins, albumin, albumin_and_globulin_ratio, ggt_value, prot_value, prediction_result) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            values = (name, age, sex, total_bilirubin, direct_bilirubin, alkaline_phosphatase, alamine_aminotransferase, aspartate_aminotransferase, total_proteins, albumin, albumin_and_globulin_ratio, ggt_value, prot_value, prediction_result)

            cursor.execute(sql_query, values)
            connection.commit()
            print("Data inserted into hepatitis_predictions table in SQLite successfully")

        except sqlite3.Error as e:
            print(f"Error: {e}")

        finally:
            if connection:
                cursor.close()
                connection.close()
                print("SQLite connection closed")

    # Call the function to create the table for hepatitis predictions
    create_hepatitis_table()
    # Hepatitis prediction page
    if selected == 'Hepatitis Prediction':
        st.title("Hepatitis Prediction")
        image = Image.open('images/hepatite.png')
        st.image(image, caption='Hepatitis Prediction')
        st.subheader('Hepatitis Symptom Ranges 📊')
        st.write('Total Bilirubin Range: 1.2 to 2.5 mg/dL')
        st.write('Direct Bilirubin Range: > 0.3 mg/dL')
        st.write('Alkaline Phosphatase: 0.73 to 2.45 microkatal per liter (µkat/L)')
        st.write('Alamine Aminotransferase : 30 to 100 IU/mL')
        st.write('Aspartate Aminotransferase: < 250 IU/mL')
        st.write('Total Proteins: 6 to 8 g/dL')
        st.write('Albumin: 3.4 to 5.4 g/dL')
        st.write('Albumin and Globulin Ratio: < 1.0')
        st.write('GGT value: 0 to 30 IU/L')
        st.write('PROT value: 6 to 8 g/dL')
        
        st.write('Note: The above values are taken for only prediction purpose not exact values !')

        # Columns
        # No inputs from the user
        name = st.text_input("Name:")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Enter your age")  # 2
        with col2:
            sex = st.selectbox("Gender", ["Male", "Female"])
            sex = 1 if sex == "Male" else 2
        with col3:
            total_bilirubin = st.number_input("Enter your Total Bilirubin")  # 3

        with col1:
            direct_bilirubin = st.number_input("Enter your Direct Bilirubin")  # 4
        with col2:
            alkaline_phosphatase = st.number_input("Enter your Alkaline Phosphatase")  # 5
        with col3:
            alamine_aminotransferase = st.number_input("Enter your Alamine Aminotransferase")  # 6

        with col1:
            aspartate_aminotransferase = st.number_input("Enter your Aspartate Aminotransferase")  # 7
        with col2:
            total_proteins = st.number_input("Enter your Total Proteins")  # 8
        with col3:
            albumin = st.number_input("Enter your Albumin")  # 9

        with col1:
            albumin_and_globulin_ratio = st.number_input("Enter your Albumin and Globulin Ratio")  # 10

        with col2:
            your_ggt_value = st.number_input("Enter your GGT value")  # Add this line
        with col3:
            your_prot_value = st.number_input("Enter your PROT value")  # Add this line

        # Code for prediction
        hepatitis_result = ''

        # Button
        if st.button("Predict Hepatitis"):
            # Create a DataFrame with user inputs
            user_data = pd.DataFrame({
                'Age': [age],
                'Sex': [sex],
                'ALB': [total_bilirubin],  # Correct the feature name
                'ALP': [direct_bilirubin],  # Correct the feature name
                'ALT': [alkaline_phosphatase],  # Correct the feature name
                'AST': [alamine_aminotransferase],
                'BIL': [aspartate_aminotransferase],  # Correct the feature name
                'CHE': [total_proteins],  # Correct the feature name
                'CHOL': [albumin],  # Correct the feature name
                'CREA': [albumin_and_globulin_ratio],  # Correct the feature name
                'GGT': [your_ggt_value],  # Replace 'your_ggt_value' with the actual value
                'PROT': [your_prot_value]  # Replace 'your_prot_value' with the actual value
            })

            # Perform prediction
            hepatitis_prediction = hepatitis_model.predict(user_data)
            # Display result
            if hepatitis_prediction[0] == 1:
                hepatitis_result = "We are really sorry to say, The model predicts that seems like you have Hepatitis."
                insert_hepatitis_data_sqlite(name, age, sex, total_bilirubin, direct_bilirubin, alkaline_phosphatase, alamine_aminotransferase, aspartate_aminotransferase, total_proteins, albumin, albumin_and_globulin_ratio, your_ggt_value, your_prot_value, "Hepatitis")
                image = Image.open('positive.png')
                st.image(image, caption='')
            else:
                hepatitis_result = 'Congratulations, The model predicts that you do not have Hepatitis.'
                insert_hepatitis_data_sqlite(name, age, sex, total_bilirubin, direct_bilirubin, alkaline_phosphatase, alamine_aminotransferase, aspartate_aminotransferase, total_proteins, albumin, albumin_and_globulin_ratio, your_ggt_value, your_prot_value, "No Hepatitis")
                image = Image.open('negative.png')
                st.image(image, caption='')

            st.success(name + ', ' + hepatitis_result)

    from sklearn.preprocessing import LabelEncoder
    import joblib
    # Connect to SQLite and create a table
    def create_kidney_table():
        connection = None
        try:
            connection = sqlite3.connect('kidney_predictions.db')
            cursor = connection.cursor()

            # Define the table schema
            create_table_query = '''
                CREATE TABLE IF NOT EXISTS kidney_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    age INTEGER,
                    bp INTEGER,
                    sg REAL,
                    al INTEGER,
                    su INTEGER,
                    rbc INTEGER,
                    pc INTEGER,
                    pcc INTEGER,
                    ba INTEGER,
                    bgr INTEGER,
                    bu INTEGER,
                    sc INTEGER,
                    sod INTEGER,
                    pot INTEGER,
                    hemo INTEGER,
                    pcv INTEGER,
                    wc INTEGER,
                    rc INTEGER,
                    htn INTEGER,
                    dm INTEGER,
                    cad INTEGER,
                    appet INTEGER,
                    pe INTEGER,
                    ane INTEGER,
                    prediction_result TEXT
                )
            '''

            # Execute the query to create the table
            cursor.execute(create_table_query)
            connection.commit()

            print("Table 'kidney_predictions' created successfully in SQLite database")

        except sqlite3.Error as e:
            print(f"Error: {e}")

        finally:
            if connection:
                connection.close()
                print("SQLite connection closed")

    # Function to insert data into SQLite
    def insert_kidney_data_sqlite(name, age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane, prediction_result):
        connection = sqlite3.connect('kidney_predictions.db')
        cursor = connection.cursor()

        try:
            sql_query = "INSERT INTO kidney_predictions (name, age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane, prediction_result) " \
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            values = (name, age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane, prediction_result)

            cursor.execute(sql_query, values)
            connection.commit()
            print("Data inserted into kidney_predictions table in SQLite successfully")

        except sqlite3.Error as e:
            print(f"Error: {e}")

        finally:
            if connection:
                cursor.close()
                connection.close()
                print("SQLite connection closed")

    # Call the function to create the table for kidney predictions
    create_kidney_table()

    # Chronic Kidney Disease Prediction Page
    if selected == 'Chronic Kidney Prediction':
        st.title("Chronic Kidney Disease Prediction")
        image = Image.open('images/kidney1.jpg')
        st.image(image, caption='Chronic Kidney Disease')
        # Add the image for Chronic Kidney Disease prediction if needed
        name = st.text_input("Name:")
        # Columns
        # No inputs from the user
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Enter your age", 1, 100, 25)  # 2
        with col2:
            bp = st.slider("Enter your Blood Pressure", 50, 200, 120)  # Add your own ranges
        with col3:
            sg = st.slider("Enter your Specific Gravity", 1.0, 1.05, 1.02)  # Add your own ranges

        with col1:
            al = st.slider("Enter your Albumin", 0, 5, 0)  # Add your own ranges
        with col2:
            su = st.slider("Enter your Sugar", 0, 5, 0)  # Add your own ranges
        with col3:
            rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
            rbc = 1 if rbc == "Normal" else 0

        with col1:
            pc = st.selectbox("Pus Cells", ["Normal", "Abnormal"])
            pc = 1 if pc == "Normal" else 0
        with col2:
            pcc = st.selectbox("Pus Cell Clumps", ["Present", "Not Present"])
            pcc = 1 if pcc == "Present" else 0
        with col3:
            ba = st.selectbox("Bacteria", ["Present", "Not Present"])
            ba = 1 if ba == "Present" else 0

        with col1:
            bgr = st.slider("Enter your Blood Glucose Random", 50, 200, 120)  # Add your own ranges
        with col2:
            bu = st.slider("Enter your Blood Urea", 10, 200, 60)  # Add your own ranges
        with col3:
            sc = st.slider("Enter your Serum Creatinine", 0, 10, 3)  # Add your own ranges

        with col1:
            sod = st.slider("Enter your Sodium", 100, 200, 140)  # Add your own ranges
        with col2:
            pot = st.slider("Enter your Potassium", 2, 7, 4)  # Add your own ranges
        with col3:
            hemo = st.slider("Enter your Hemoglobin", 3, 17, 12)  # Add your own ranges

        with col1:
            pcv = st.slider("Enter your Packed Cell Volume", 20, 60, 40)  # Add your own ranges
        with col2:
            wc = st.slider("Enter your White Blood Cell Count", 2000, 20000, 10000)  # Add your own ranges
        with col3:
            rc = st.slider("Enter your Red Blood Cell Count", 2, 8, 4)  # Add your own ranges

        with col1:
            htn = st.selectbox("Hypertension", ["Yes", "No"])
            htn = 1 if htn == "Yes" else 0
        with col2:
            dm = st.selectbox("Diabetes Mellitus", ["Yes", "No"])
            dm = 1 if dm == "Yes" else 0
        with col3:
            cad = st.selectbox("Coronary Artery Disease", ["Yes", "No"])
            cad = 1 if cad == "Yes" else 0

        with col1:
            appet = st.selectbox("Appetite", ["Good", "Poor"])
            appet = 1 if appet == "Good" else 0
        with col2:
            pe = st.selectbox("Pedal Edema", ["Yes", "No"])
            pe = 1 if pe == "Yes" else 0
        with col3:
            ane = st.selectbox("Anemia", ["Yes", "No"])
            ane = 1 if ane == "Yes" else 0

        # Code for prediction
        kidney_result = ''

        # Button
        if st.button("Predict Chronic Kidney Disease"):
            # Create a DataFrame with user inputs
            user_input = pd.DataFrame({
                'age': [age],
                'bp': [bp],
                'sg': [sg],
                'al': [al],
                'su': [su],
                'rbc': [rbc],
                'pc': [pc],
                'pcc': [pcc],
                'ba': [ba],
                'bgr': [bgr],
                'bu': [bu],
                'sc': [sc],
                'sod': [sod],
                'pot': [pot],
                'hemo': [hemo],
                'pcv': [pcv],
                'wc': [wc],
                'rc': [rc],
                'htn': [htn],
                'dm': [dm],
                'cad': [cad],
                'appet': [appet],
                'pe': [pe],
                'ane': [ane]
            })

            # Perform prediction
            kidney_prediction = chronic_disease_model.predict(user_input)
            # Display result
            if kidney_prediction[0] == 1:
                kidney_prediction_dig = "we are really sorry to say, The model predicts that seems like you have kidney disease."
                insert_kidney_data_sqlite(name, age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane, "Chronic Kidney Disease")
                image = Image.open('positive.png')
                st.image(image, caption='')
                
            else:
                image = Image.open('negative.png')
                st.image(image, caption='')
                kidney_prediction_dig = "Congratulation, The model predicts that You don't have kidney disease."
                insert_kidney_data_sqlite(name, age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane, "No Chronic Kidney Disease")
            st.success(name+' , ' + kidney_prediction_dig)
    if selected == 'Classification Reports':
        with st.container():
            st.title('Classification Reports')
            selected1 = option_menu(
                menu_title=None,
                options=['Diabetes','Heart-Disease','Hepatitis','Lung Cancer','Chronic Kidney'],
                icons=['activity','heart','','lungs','person'],
                orientation='horizontal'
            )
            if selected1=='Diabetes':
                st.subheader('Classification Report for Diabetes Disease Prediction')
                image = Image.open('images/diab3.png')
                st.image(image, caption='Classification Report for Bagging SVM')
                image = Image.open('images/diab4.png')
                st.image(image, caption='Classification Report for AdaBoost SVM')
                image = Image.open('images/diab1.png')
                st.image(image, caption='Model Comparison in Bar chart')
                image = Image.open('images/diab2.png')
                st.image(image, caption='Model Comparison in Pie chart')
                
            if selected1=='Heart-Disease':
                st.subheader('Classification Report for Heart Disease Prediction')
                image = Image.open('images/heart1.png')
                st.image(image, caption='Classification Report for Logistic Regression and KNN')
                image = Image.open('images/heart2.png')
                st.image(image, caption='Classification Report for Random Forest and SVM')
                st.write('Logistic Regression: 0.7951219512195122')
                st.write('KNN: 0.7317073170731707')
                st.write('Random Forest: 0.9853658536585366')
                st.write('SVM: 0.6829268292682927')          
                image = Image.open('images/heart3.png')
                st.image(image, caption='Model Comparison in Bar chart')
                image = Image.open('images/heart4.png')
                st.image(image, caption='Model Comparison in Pie chart')
                
            if selected1=='Hepatitis':
                st.subheader('Classification Report for Hepatitis Disease Prediction')
                image = Image.open('images/hepa1.png')
                st.image(image, caption='Classification Report for Logistic Regression')
                image = Image.open('images/hepa3.png')
                st.image(image, caption='Classification Report for Random Forest')
                image = Image.open('images/hepa4.png')
                st.image(image, caption='Classification Report for Support Vector Machine')
                image = Image.open('images/hepa2.png')
                st.image(image, caption='Model Comparison in Bar chart')
            
            if selected1=='Lung Cancer':
                st.subheader('Classification Report for Lung Cancer Disease Prediction')
                image = Image.open('images/lung1.png')
                st.image(image, caption='Classification Report for Logistic Regression')
                image = Image.open('images/lung2.png')
                st.image(image, caption='Confusion Matrix for Logistic Regression')
            
            if selected1=='Chronic Kidney':
                st.subheader('Classification Report for Chronic Kidney Disease Prediction')
                image = Image.open('images/kidney1.png')
                st.image(image, caption='Classification Report for Logistic Regression')
                image = Image.open('images/kidney2.png')
                st.image(image, caption='Classification Report for SVM')
                image = Image.open('images/kidney3.png')
                st.image(image, caption='Model Comparison in Bar chart')
                image = Image.open('images/kidney4.png')
                st.image(image, caption='Confusion Matrix for Logistic Regression')
                image = Image.open('images/kidney5.png')
                st.image(image, caption='Confusion Matrix for SVM')
    if selected == 'Contact us':
        st.header(":mailbox: Get In Touch To Share your Ideas With Us!")
        contact_form = """
        <form action="https://formspree.io/f/xleqbddl" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your name" required>
            <input type="email" name="email" placeholder="Your email" required>
            <textarea name="message" placeholder="Your message here"></textarea>
            <button type="submit">Send</button>
        </form>
        """

        st.markdown(contact_form, unsafe_allow_html=True)

        # Use Local CSS File
        def local_css(file_name):
            with open(file_name) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        local_css("style.css")
        
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

def set_bg_from_url(url, opacity=1):
    # Set background image using HTML and CSS
    st.markdown(
        f"""
        <style>
            body {{
                background: url('{url}') no-repeat center center fixed;
                background-size: cover;
                opacity: {opacity};
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image from URL
set_bg_from_url("https://cdn.pixabay.com/photo/2018/07/15/10/44/dna-3539309_1280.jpg", opacity=0.875)