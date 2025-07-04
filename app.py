import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# --- Cached model loading ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.keras')

# --- Cached pickle loading ---
@st.cache_resource
def load_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

# Load model and preprocessing objects
model = load_model()
label_encoder_gender = load_pickle('label_encoder_gender.pkl')
onehot_encoder_geo = load_pickle('onehot_encoder_geo.pkl')
scaler = load_pickle('scaler.pkl')

# --- Streamlit UI ---
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# --- Prepare input data ---
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Encode geography
geo_encoded = onehot_encoder_geo.transform([[geography]])
geo_df = pd.DataFrame(geo_encoded.toarray(), columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine all features
full_input = pd.concat([input_data, geo_df], axis=1)

# Scale input
scaled_input = scaler.transform(full_input)

# Predict
prediction = model.predict(scaled_input)
churn_prob = prediction[0][0]

# --- Output result ---
st.write(f"**Churn Probability:** `{churn_prob:.2f}`")

if churn_prob > 0.5:
    st.error("⚠️ The customer is likely to churn.")
else:
    st.success("✅ The customer is not likely to churn.")
