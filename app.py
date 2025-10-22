import streamlit as st 
import joblib

model = joblib.load("spam_email_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

#giving title to the web app 
st.title ("Spam Email Detector")
st.write("Welcome to the Spam Email Detector App! This application uses a machine learning model to classify emails as 'Spam' or 'Not Spam' (Ham). Simply enter the content of an email below, and the app will predict whether it's spam or not.")

#taking input email from the user
input_email = st.text_area("Enter the email content here:")
if st.button("PREDICT"):
    if input_email.strip() == "":
        st.warning("Please enter the email content to proceed with the prediction.")
    else:
        input_email_vector = vectorizer.transform([input_email])
        prediction = model.predict(input_email_vector)
        if prediction[0] == 1:
            st.error("The email is classified as: SPAM🚨")
        else:
            st.success("The email is classified as: NOT SPAM (HAM)🟢")