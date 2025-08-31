import joblib
from huggingface_hub import HfApi, create_repo, upload_file
import os

# Define your model repository ID on Hugging Face
model_repo_id = "LalithaShiva/tourism_prediction_model" # Replace with your desired model repository ID

# Define the path to your saved model file
model_save_path = "best_random_forest_model.joblib" # Replace with the actual path to your saved model

# Initialize the HfApi
api = HfApi()

# 1. Create the model repository on the Hugging Face Hub (if it doesn't exist)
try:
    create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True) # Use exist_ok=True
    print(f"Model repository '{model_repo_id}' created or already exists on Hugging Face.")
except Exception as e:
    print(f"Could not create model repository: {e}")


# 2. Upload the saved model file to the repository
if os.path.exists(model_save_path):
    try:
        upload_file(
            path_or_fileobj=model_save_path,
            path_in_repo="best_model.joblib", # The name the file will have in the repo
            repo_id=model_repo_id,
            repo_type="model",
        )
        print(f"Best model uploaded to Hugging Face Hub: {model_repo_id}")
    except Exception as e:
        print(f"Error uploading model to Hugging Face: {e}")
else:
    print(f"Error: Model file not found at {model_save_path}. Skipping upload.")

# 3. Upload any associated preprocessing files (e.g., LabelEncoders, scalers)
# If you saved any preprocessing objects, uncomment and modify this section
# example_preprocessor_path = "label_encoder.joblib" # Replace with the path to your saved preprocessor
# if os.path.exists(example_preprocessor_path):
#     try:
#         upload_file(
#             path_or_fileobj=example_preprocessor_path,
#             path_in_repo="label_encoder.joblib", # The name the file will have in the repo
#             repo_id=model_repo_id,
#             repo_type="model",
#         )
#         print(f"Preprocessor file uploaded to Hugging Face Hub: {model_repo_id}")
#     except Exception as e:
#         print(f"Error uploading preprocessor file to Hugging Face: {e}")
# else:
#     print(f"Warning: Preprocessor file not found at {example_preprocessor_path}. Skipping upload.")

print("\nModel registration process finished.")