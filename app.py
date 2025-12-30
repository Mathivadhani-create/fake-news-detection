import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title("📰 Fake News Detection App")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload CSV file with 'text' and 'label' columns", type="csv"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Info")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    st.write("Column types:")
    st.write(df.dtypes)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Model training
    st.subheader("Train Fake News Classifier")
    if st.button("Train Model"):
        if "text" not in df.columns or "label" not in df.columns:
            st.error("CSV must contain 'text' and 'label' columns")
        else:
            X = df["text"]
            y = df["label"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            vectorizer = TfidfVectorizer(stop_words="english")
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            model = LogisticRegression()
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)

            acc = accuracy_score(y_test, y_pred)
            st.success(f"Model trained! Accuracy: {acc:.2f}")

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # Single prediction
            st.subheader("Test a News Article")
            user_input = st.text_area("Enter news text here")
            if st.button("Predict"):
                if user_input.strip() != "":
                    vec_input = vectorizer.transform([user_input])
                    prediction = model.predict(vec_input)[0]
                    st.write(f"Prediction: **{prediction}**")
                else:
                    st.warning("Please enter some text to predict")
else:
    st.info("Upload a CSV file to get started.")
