import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Function to load CSV file
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.lower()  # Standardize column names
        return df
    return None

# Preprocess data
def preprocess_data(df):
    scaler = StandardScaler()
    numeric_cols = ["age", "weight", "height", "steps"]
    
    if not all(col in df.columns for col in numeric_cols):
        missing_cols = set(numeric_cols) - set(df.columns)
        st.error(f"Missing columns in CSV: {missing_cols}")
        return None, None

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler

# Train the model
def train_model(df):
    required_cols = {"age", "weight", "height", "steps", "calories_burned"}
    
    if not required_cols.issubset(df.columns):
        missing_cols = required_cols - set(df.columns)
        st.error(f"Missing required columns: {missing_cols}")
        return None

    X = df[["age", "weight", "height", "steps"]]
    y = df["calories_burned"]

    if X.shape[0] == 0:
        st.error("No data available to train the model. Please upload a valid dataset.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    st.success(f"Model trained successfully! MAE: {mae:.2f}")

    joblib.dump(model, "calorie_predictor.pkl")
    return model

# Generate fitness recommendations
def generate_recommendations(steps, calories):
    if steps < 5000:
        return "Increase daily steps to at least 7000 for better health."
    elif calories < 2000:
        return "Consider increasing protein intake and strength training."
    else:
        return "You're on track! Maintain consistency."

# Streamlit UI
st.title("Personal Fitness Tracker")
st.write("Upload a CSV file (columns: `age`, `weight`, `height`, `steps`, `calories_burned`)")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.write("**Preview of Uploaded Data:**")
        st.dataframe(df)

        if "calorie_predictor.pkl" not in os.listdir():
            st.warning("No trained model found. Training a new model...")
            model = train_model(df)
        else:
            model = joblib.load("calorie_predictor.pkl")

        if model:
            st.sidebar.header("Enter Your Fitness Data")
            age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=25)
            weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
            height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=175)
            steps = st.sidebar.number_input("Daily Steps", min_value=0, max_value=30000, value=5000)

            if st.sidebar.button("Predict Calories Burned"):
                input_data = pd.DataFrame([[age, weight, height, steps]], columns=["age", "weight", "height", "steps"])
                processed_data, _ = preprocess_data(input_data)
                
                if processed_data is not None:
                    calories_pred = model.predict(processed_data)
                    st.sidebar.write(f" **Predicted Calories Burned:** {calories_pred[0]:.2f}")

                    recommendation = generate_recommendations(steps, calories_pred[0])
                    st.sidebar.write(f" **Recommendation:** {recommendation}")
