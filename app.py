import streamlit as st
import pickle

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .stApp {
        background: linear-gradient(to right, #667eea, #764ba2);
        color: white;
    }
    .stTextArea textarea {
        background-color: #ffffff;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Fake News Detector")

user_input = st.text_area("Enter News Text")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        vect = vectorizer.transform([user_input])
        prediction= model.predict(vect)
        st.write("Raw Prediction:", prediction[0])
        st.write("Type:", type(prediction[0]))
        st.subheader("Predict Result")

        if prediction[0] == "REAL":
            st.success("Predict: REAL 🟢")
        else:
            st.error("Predict: FAKE 🔴")

        st.subheader("Model Performance")
        st.image("confusion_matrix.png")