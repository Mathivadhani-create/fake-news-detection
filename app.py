import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

st.set_page_config(page_title="Fake News Detection", layout="wide")

st.title("📰 Fake News Detection App")
st.write(
    "Upload a dataset with **text** and **label** columns to train a "
    "fake-news classifier. The model uses TF-IDF + Logistic Regression."
)

uploaded_file = st.file_uploader(
    "Upload CSV file (columns: text, label)",
    type=["csv"],
)

if uploaded_file:

    # -----------------------------
    # Load and preview data
    # -----------------------------
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if "text" not in df.columns or "label" not in df.columns:
        st.error(
            "❌ The CSV must contain two columns: **text** and **label**"
        )
    else:
        st.success("✔ Data looks good!")

        # -----------------------------
        # Train / Test Split
        # -----------------------------
        X = df["text"]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
        )

        # -----------------------------
        # Text Vectorization (TF-IDF)
        # -----------------------------
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # -----------------------------
        # Model Training
        # -----------------------------
        model = LogisticRegression(max_iter=200)
        model.fit(X_train_vec, y_train)

        # -----------------------------
        # Evaluation
        # -----------------------------
        y_pred = model.predict(X_test_vec)

        st.subheader("Model Performance")

        col1, col2 = st.columns(2)

        with col1:
            accuracy = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{accuracy:.2%}")

            st.text("Classification Report")
            st.code(classification_report(y_test, y_pred))

        with col2:
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots()
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
            )
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        # -----------------------------
        # Real-time Prediction
        # -----------------------------
        st.subheader("🔎 Try Your Own Text")

        user_text = st.text_area(
            "Enter a news headline or article:"
        )

        if st.button("Predict"):
            if user_text.strip() == "":
                st.warning("Please type some text first.")
            else:
                vec = vectorizer.transform([user_text])
                pred = model.predict(vec)[0]

                if pred == 1 or pred == "FAKE":
                    st.error("🚨 Likely FAKE news")
                else:
                    st.success("✅ Likely REAL news")

else:
    st.info("Upload a CSV file to get started.")
