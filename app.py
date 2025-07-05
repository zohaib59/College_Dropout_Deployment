import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

st.set_page_config(page_title="College Dropout Prediction", layout="wide")

DATA_PATH = "dropout_2.csv"

# Load or fallback to sample data
@st.cache_data
def load_data_and_train():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, encoding='Latin1')
    else:
        st.error("❌ dropout_2.csv not found.")
        st.stop()

    # Label encode all object columns
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop("Application_Status", axis=1)
    y = df["Application_Status"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(verbosity=0, n_jobs=-1, use_label_encoder=False)
    model.fit(X_train, y_train)

    return df, model, scaler, label_encoders, X_train, X_test, y_train, y_test

# Load model and data
df, model, scaler, label_encoders, X_train, X_test, y_train, y_test = load_data_and_train()

# UI
st.title("🎓 College Dropout Prediction App")

menu = st.sidebar.radio("Choose Option", ["📊 Show Dataset", "🔍 Predict Dropout", "📈 Model Evaluation"])

# Show Dataset
if menu == "📊 Show Dataset":
    st.subheader("📌 Dataset Preview")
    st.dataframe(df)

# Predict New Data
elif menu == "🔍 Predict Dropout":
    st.subheader("🔎 Enter Student Details to Predict Dropout Risk")

    with st.form("predict_form"):
        input_data = {}

        for col in df.drop("Application_Status", axis=1).columns:
            if col in label_encoders:
                input_data[col] = st.selectbox(col, label_encoders[col].classes_)
            else:
                input_data[col] = st.number_input(
                    label=col,
                    value=float(df[col].mean())
                )

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([input_data])

        # Encode using label encoders
        for col in input_df.columns:
            if col in label_encoders:
                input_df[col] = label_encoders[col].transform([input_df[col]])  # convert category to code

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        prediction_label = label_encoders["Application_Status"].inverse_transform([prediction])[0]

        st.success(f"🎯 **Predicted Application Status:** {prediction_label}")

# Evaluation Page
elif menu == "📈 Model Evaluation":
    st.subheader("📉 Model Performance on Test Data")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    st.write(f"✅ **Accuracy:** {acc:.2f}")
    st.text("📋 Classification Report:")
    st.text(report)
