
# 💼 Customer Churn Prediction

This project uses a trained **Artificial Neural Network (ANN)** to predict whether a customer will **churn (leave)** a bank. It includes a user-friendly **Streamlit web app** where users can input customer data and get predictions.

## 🚀 Features

- Trained with `Churn_Modelling.csv` dataset
- Encodes categorical data (Gender, Geography)
- Scales numerical features
- Uses a multi-layer ANN with `TensorFlow/Keras`
- Streamlit web app for real-time predictions
- Pickled encoders and scalers for consistent preprocessing

## 🧠 Tech Stack

- Python
- TensorFlow / Keras
- Pandas, NumPy
- Scikit-learn (LabelEncoder, OneHotEncoder, StandardScaler)
- Streamlit
- Pickle

## 📁 Project Structure

```
├── app.py                       # Streamlit app
├── model.keras                 # Trained ANN model
├── label_encoder_gender.pkl    # Gender LabelEncoder
├── onehot_encoder_geo.pkl      # Geography OneHotEncoder
├── scaler.pkl                  # StandardScaler
├── Churn_Modelling.csv         # Dataset (optional for training)
```

## 📊 Model Architecture

- Input: 13 features (after encoding)
- Hidden Layer 1: 64 neurons, ReLU
- Hidden Layer 2: 32 neurons, ReLU
- Output Layer: 1 neuron, Sigmoid
- Loss: Binary Crossentropy
- Optimizer: Adam

## 🔧 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/your-username/churn-prediction-app.git
cd churn-prediction-app
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the app**
```bash
streamlit run app.py
```

4. **Interact with the UI**
Open in your browser → http://localhost:8501

## 📦 Requirements (`requirements.txt`)

```txt
streamlit
tensorflow
scikit-learn
pandas
numpy
```

## 📝 How It Works

1. User selects values for age, salary, geography, etc.
2. Input is encoded and scaled just like training.
3. The ANN predicts the churn probability.
4. Based on threshold (0.5), a message is shown.

## ✅ Example Prediction

> Input:
```
Age: 40, Geography: France, Gender: Male,
CreditScore: 600, Balance: 60,000, EstimatedSalary: 50,000, Tenure: 3
```

> Output:
```
Churn Probability: 0.09
✅ The customer is not likely to churn.
```

## 📌 Notes

- The model and encoders must match training pipeline exactly.
- Use `@st.cache_resource` to avoid repeated model/encoder loading in Streamlit.

## 📚 Credits

Data Source: [Churn Modelling Dataset](https://www.kaggle.com/datasets/shubhendra7/customer-churn-prediction)
