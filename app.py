import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="💳",
    layout="centered"
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #45a049;
    }

    .title {
        text-align: center;
        color: #2E86C1;
        font-size: 40px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        color: gray;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-good {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        color: #155724;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
    }

    .result-bad {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 10px;
        color: #721c24;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

model = joblib.load('xgb_model.pkl')

st.markdown(
    '<p class="title">💳 Credit Score Prediction</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Predict whether a customer is Good Risk or Bad Risk</p>',
    unsafe_allow_html=True
)

st.subheader("📋 Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        'Age',
        min_value=18,
        max_value=100
    )

    sex = st.selectbox(
        'Sex',
        ['male', 'female']
    )

    job = st.selectbox(
        'Job Level',
        [0, 1, 2, 3]
    )

    housing = st.selectbox(
        'Housing',
        ['own', 'rent', 'free']
    )

with col2:
    saving_accounts = st.selectbox(
        'Saving Accounts',
        ['little', 'moderate', 'rich']
    )

    checking_account = st.selectbox(
        'Checking Account',
        ['little', 'moderate', 'rich']
    )

    credit_amount = st.number_input(
        'Credit Amount',
        min_value=0,
        value=100
    )

    duration = st.number_input(
        'Duration (Months)',
        min_value=0,
        value=12
    )

if st.button('🔍 Predict Credit Risk'):

    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Job': [job],
        'Housing': [housing],
        'Saving accounts': [saving_accounts],
        'Checking account': [checking_account],
        'Credit amount': [credit_amount],
        'Duration': [duration]
    })

    cat_cols = [
        'Sex',
        'Housing',
        'Saving accounts',
        'Checking account'
    ]

    for col in cat_cols:
        le = joblib.load(f'{col}_encoder.pkl')
        input_data[col] = le.transform(input_data[col])

    prediction = model.predict(input_data)

    st.markdown("---")

    if prediction[0] == 1:
        st.markdown(
            '<div class="result-good">✅ Good Risk Customer</div>',
            unsafe_allow_html=True
        )
        st.balloons()

    else:
        st.markdown(
            '<div class="result-bad">❌ Bad Risk Customer</div>',
            unsafe_allow_html=True
        )