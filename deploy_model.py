import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download, HfApi, create_repo, upload_file

# --- Deployment Steps ---

# 1. Define a Dockerfile and list all configurations
# Note: Dockerfile content is provided separately. You need to create a file named 'Dockerfile'
# with the content provided in a previous Markdown cell and ensure it's in the same directory
# as this script and other deployment files.

# 2. Load the saved model from the Hugging Face model hub
print("Loading model from Hugging Face Hub...")
model_repo_id = "LalithaShiva/tourism_prediction_model" # Replace with your actual model repository ID
model_filename = "best_model.joblib" # Replace with the actual filename of your saved model

try:
    model_path = hf_hub_download(repo_id=model_repo_id, filename=model_filename)
    print(f"Model downloaded to: {model_path}")
    loaded_model = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model from Hugging Face Hub: {e}")
    loaded_model = None


# 3. Get the inputs and save them into a dataframe
# This part is highly dependent on how your application receives inputs (e.g., API, user interface)
# You need to replace this with your actual code to get input data and create a pandas DataFrame
print("\nGetting inputs...")
# Example: Create a dummy input DataFrame (replace with your actual input handling)
if loaded_model:
    # Create a dummy DataFrame with the same columns as your training data (excluding the target)
    # You will need to ensure the column names and data types match your model's expectations
    # based on your data preparation steps (including encoding categorical features).
    # This dummy data structure should reflect the expected input format for your model.
    # For a real application, you would get this data from a user request, file, etc.

    # Assuming X_train_columns is a list of the feature column names from your training data
    try:
        # This assumes X_train is available from previous steps and has the correct columns
        input_data_structure = {col: [0] for col in X_train.columns} # Replace with actual sample data or input mechanism
        input_df = pd.DataFrame(input_data_structure)
        print("Dummy input DataFrame created. REPLACE WITH YOUR ACTUAL INPUT HANDLING.")
        print(input_df.head())

        # --- Perform the same preprocessing as on training data ---
        # This is CRUCIAL. You need to apply the same steps (imputation, encoding, etc.)
        # to the input_df as you did to X_train.
        # If you used LabelEncoding, you need to load the fitted LabelEncoders and apply them.
        # If you used imputation, you need the imputation values (median/mode from training)
        # and apply them.
        print("\nApplying preprocessing to input data...")
        # Add your preprocessing code here. Example placeholders:
        # for col in input_df.columns:
        #     if col in training_label_encoders: # Check if column was label encoded
        #         input_df[col] = training_label_encoders[col].transform(input_df[col])
        #     elif col in training_median_values: # Check if numerical and needs imputation
        #          input_df[col].fillna(training_median_values[col], inplace=True)
        #     elif col in training_mode_values: # Check if categorical and needs imputation
        #          input_df[col].fillna(training_mode_values[col], inplace=True)
        print("Preprocessing applied (replace with your actual preprocessing logic).")


    except NameError:
        print("Error: X_train is not available. Cannot create dummy input DataFrame. Ensure X_train is defined or define input_df manually.")
        input_df = None
    except Exception as e:
        print(f"Error creating or preprocessing input DataFrame: {e}")
        input_df = None
else:
    input_df = None
    print("Skipping input handling: Model not loaded.")


# 4. Define a dependencies file for the deployment
# This creates the requirements.txt file. Ensure all necessary libraries are listed.
print("\nCreating requirements.txt...")
requirements_content = """
scikit-learn
pandas
huggingface_hub
joblib
streamlit # Example: if using Streamlit for your app
""" # Add any other libraries your app.py needs

with open("requirements.txt", "w") as f:
    f.write(requirements_content)
print("requirements.txt file created.")


# 5. Define a hosting script that can push all the deployment files into the Hugging Face space
# This is the script that uploads your Dockerfile, requirements.txt, model file, and app.py
print("\nGenerating upload script content...")
upload_script_content = """
# Save this code as a Python file (e.g., upload_to_space.py)

from huggingface_hub import HfApi, create_repo, upload_file
import os

# Define your Hugging Face Space repository ID
space_repo_id = "LalithaShiva/tourismSalesPrediction" # Replace with your desired Space repository ID

# Initialize the HfApi
api = HfApi()

# Define the files you want to upload
files_to_upload = [
    "Dockerfile",
    "requirements.txt",
    "best_random_forest_model.joblib", # Your saved model file
    "app.py" # Your inference script
]

# Create the Space repository on the Hugging Face Hub (if it doesn't exist)
try:
    create_repo(repo_id=space_repo_id, repo_type="space", space_sdk="docker", exist_ok=True) # Use exist_ok=True
    print(f"Space repository '{space_repo_id}' created or already exists on Hugging Face.")
except Exception as e:
    print(f"Could not create Space repository: {e}")


# Upload the files to the Space repository
print(f"\nUploading files to {space_repo_id}...")
for file_path in files_to_upload:
    if os.path.exists(file_path):
        try:
            upload_file(
                path_or_fileobj=file_path,
                path_in_repo=os.path.basename(file_path),
                repo_id=space_repo_id,
                repo_type="space",
            )
            print(f"Uploaded {file_path}")
        except Exception as e:
            print(f"Error uploading {file_path}: {e}")
    else:
        print(f"Error: File not found - {file_path}. Skipping upload.")

print("\nUpload process finished.")

# Remember to create your app.py file with the inference logic before running this script.
# This script assumes you have already logged in to Hugging Face using notebook_login()
# or have your token set as an environment variable.
"""

print("--- upload_to_space.py Content ---")
print(upload_script_content)
print("----------------------------------")

# You can save this content to a file named 'upload_to_space.py' and run it.
# Example of saving the upload script content to a file:
# with open("upload_to_space.py", "w") as f:
#     f.write(upload_script_content)
# print("\n'upload_to_space.py' file created. Run this script to upload your deployment files.")

