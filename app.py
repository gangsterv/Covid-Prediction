import streamlit as st
import pandas as pd
import joblib

def app():
    model = joblib.load('model.h5')
    st.set_page_config(page_title="Covid Prediction")
    st.title("Covid Patient Prediction")
    st.header("Epsilon Diploma Project")

    st.write("This project predicts covid status based on some features")

    gender = st.radio('Select Gender', ['Male', 'Female'])
    intubed = st.radio("Intubation", ['Yes', 'No', 'Not Applicable'])
    age = st.number_input("Age", value=0)
    alive = st.radio('Alive?', ['Yes', 'No'])
    other_covid = st.selectbox("Has other covid?", ['Yes', 'No', 'Not Specified'])
    pneu = st.selectbox("Has Pneumonia", ['Yes', 'No'])
    hyper = st.selectbox("Has Hypertension", ['Yes', 'No'])
    obesity = st.selectbox("Has Obesity", ['Yes', 'No'])
    diabetes = st.selectbox("Has Diabetes", ['Yes', 'No'])

    predict = st.button("Predict")
    if predict:
        df = pd.DataFrame.from_dict(
            {
                'sex':[1 if gender == 'Female' else 0],
                'intubed':[1 if intubed == 'Yes' else (0 if intubed == 'No' else 97)],
                'age':[age],
                'died':[1 if alive == 'Yes' else 0],
                'contact_other_covid':[1 if other_covid == 'Yes' else (0 if other_covid == 'No' else 99)],
                'pneumonia':[1 if pneu == 'Yes' else 0],
                'hypertension':[1 if hyper == 'Yes' else 0],
                'obesity':[1 if obesity == 'Yes' else 0],
                'diabetes':[1 if diabetes == 'Yes' else 0]
            }
        )

        st.write("Input Data: ")
        st.dataframe(df)

        pred = model.predict(df)
        st.write(F"Prediction: {pred}")

app()