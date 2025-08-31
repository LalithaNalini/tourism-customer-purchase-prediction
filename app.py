import pandas as pd
import joblib
import os
import streamlit as st
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import LabelEncoder

# Define the path to the model file within the Docker container.
# This file is copied into the container by your Dockerfile.
MODEL_PATH = "best_random_forest_model.joblib"

# Load the model directly from the local file path
@st.cache_resource # Cache the model loading
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
print("Model Loaded successfully")

# Streamlit UI for Tourism Sales Prediction
st.title("Tourism Sales Prediction App")
st.write("""
This application predicts whether a customer will take a tourism package based on their details.
Please enter the customer details below to get a prediction.
""")

# User input fields based on the tourism dataset columns
st.header("Customer Details")

age = st.number_input("Age", min_value=0, max_value=120, value=30)
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, value=10.0)
number_of_person_visiting = st.number_input("NumberOf Persons Visiting", min_value=1, value=2)
number_of_followups = st.number_input("NumberOf Followups", min_value=0, value=3)
preferred_property_star = st.selectbox("Preferred Property Star", [1.0, 2.0, 3.0, 4.0, 5.0])
number_of_trips = st.number_input("Number of Trips", min_value=0, value=1)
passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
number_of_children_visiting = st.number_input("NumberOf Children Visiting", min_value=0, value=0)
monthly_income = st.number_input("Monthly Income", min_value=0.0, value=25000.0)

# Handling categorical features - Collect user input
typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer", "Other", "Housewife"])
gender = st.selectbox("Gender", ["Male", "Female", "Fe Male"]) # Note: "Fe Male" was in the original data
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])

# Assemble input into DataFrame
input_data = {
    'Age': age,
    'TypeofContact': typeofcontact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'NumberOfFollowups': number_of_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}
input_df = pd.DataFrame([input_data])

# --- Preprocessing the input data ---
# Apply the same preprocessing steps as used during training
# This includes handling missing values and encoding categorical features.
# Assuming LabelEncoding was used for categorical features during training:

# Identify categorical columns (based on the original data types before encoding)
categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'Designation', 'ProductPitched']

# Apply LabelEncoding to categorical columns in the input DataFrame
# Note: You should ideally save the fitted LabelEncoders from training and load them here
# to ensure consistent encoding. For this example, we re-fit LabelEncoder, which might
# lead to issues if the input data contains categories not seen during training.
# A robust solution involves saving and loading the encoders.
# For a robust deployment, fit LabelEncoders on the training data and save/load them.
# For demonstration purposes, re-fitting here (less ideal for production):
for col in categorical_cols:
    if col in input_df.columns:
        # Simple imputation for missing values (if any) in categorical columns
        input_df[col].fillna(input_df[col].mode()[0], inplace=True)
        # Apply Label Encoding
        # This assumes the LabelEncoder can handle the categories in the input.
        # In a real scenario, you would use the LabelEncoder fitted on the training data.
        le = LabelEncoder()
        # Fit and transform for each column - this is simplified and might need adjustment
        # if there are new categories in input not seen in training.
        # To handle unseen categories gracefully in production, consider a different encoding strategy
        # or a robust LabelEncoder implementation that handles unknown values.
        # For this example, we'll assume the input categories are seen during training.
        try:
            # This is a simplified approach; a production system should load fitted encoders
            # Create a dummy series with all possible categories seen during training
            # (you would need to get this list from your training data)
            # Example: all_training_categories = ['CatA', 'CatB', 'CatC', ...] # Load this list
            # le.fit(all_training_categories)
            # input_df[col] = le.transform(input_df[col])

            # A more robust approach without saving/loading encoders for this example:
            # Create a mapping based on all possible categories the Streamlit selectbox allows
            # This is still not ideal if training data had other categories.
            if col == 'TypeofContact':
                le.fit(["Self Enquiry", "Company Invited"])
            elif col == 'Occupation':
                 le.fit(["Salaried", "Small Business", "Large Business", "Free Lancer", "Other", "Housewife"])
            elif col == 'Gender':
                 le.fit(["Male", "Female", "Fe Male"])
            elif col == 'MaritalStatus':
                 le.fit(["Married", "Single", "Divorced", "Unmarried"])
            elif col == 'Designation':
                 le.fit(["Executive", "Manager", "Senior Manager", "AVP", "VP"])
            elif col == 'ProductPitched':
                 le.fit(["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])

            input_df[col] = le.transform(input_df[col])

        except Exception as e:
            st.error(f"Error during Label Encoding for column {col}: {e}")
            # Handle error, e.g., if an unseen category is encountered


# Handle numerical features missing values (if any)
for col in input_df.columns:
    if input_df[col].dtype != 'object':
        input_df[col].fillna(input_df[col].median(), inplace=True)

# Ensure the columns of the input DataFrame match the columns of the training data (X_train)
# This is crucial for the model to make predictions. The order and names must match.
# You should ideally save the list of column names from your training features (X_train.columns)
# and use it here to reindex the input_df.
# For this example, let's assume the expected order based on the original training data features
expected_columns = [
    'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation',
    'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups', 'ProductPitched',
    'PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips', 'Passport',
    'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'Designation',
    'MonthlyIncome'
]

if set(input_df.columns) == set(expected_columns):
    input_df = input_df[expected_columns] # Reorder columns
else:
    st.error("Input features do not match expected features. Please check preprocessing.")
    st.stop()


# --- Prediction ---
if st.button("Predict Tourism Package Purchase"):
    if model is not None and input_df is not None:
        try:
            # Ensure column order matches training data before predicting
            # This is critical if using scikit-learn models.
            # The reindexing step above should ensure the order is correct if expected_columns is accurate.

            prediction = model.predict(input_df)[0]
            result = "Customer will likely purchase a tourism package" if prediction == 1 else "Customer will likely NOT purchase a tourism package"
            st.subheader("Prediction Result:")
            st.success(f"The model predicts: **{result}**")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            # Print more details about the error for debugging
            import traceback
            st.error(traceback.format_exc())
    else:
        st.warning("Model not loaded or input data not processed correctly. Cannot make prediction.")
