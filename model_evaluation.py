import pandas as pd
import joblib
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder if used in preprocessing
import os # Import os for file existence checks


# Define your dataset and model repository IDs on Hugging Face
dataset_repo_id = "LalithaShiva/tourism_prediction_dataset" # Replace with your actual dataset repository ID
model_repo_id = "LalithaShiva/tourism_prediction_model" # Replace with your actual model repository ID
model_filename = "best_random_forest_model.joblib" # Replace with the actual filename of your saved model

# 1. Load the best model from the Hugging Face model hub
print("Loading model from Hugging Face Hub...")
try:
    # Download the model file from the Hugging Face Hub
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(repo_id=model_repo_id, filename=model_filename)
    print(f"Model downloaded to: {model_path}")
    loaded_model = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model from Hugging Face Hub: {e}")
    loaded_model = None

# 2. Load the test data from the Hugging Face data space
print("\nLoading test data from Hugging Face Hub...")
test_dataset = None
try:
    # Load only the 'test' split
    dataset = load_dataset(dataset_repo_id, split='test')
    test_dataset = dataset
    print("Test dataset loaded successfully!")
except Exception as e:
    print(f"Error loading test dataset: {e}")

if loaded_model and test_dataset:
    # Convert test dataset to pandas DataFrame
    test_df = test_dataset.to_pandas()

    # Separate features (X_test) and target (y_test)
    # Ensure 'ProdTaken' is the correct target column name
    if 'ProdTaken' in test_df.columns:
        X_test = test_df.drop("ProdTaken", axis=1)
        y_test = test_df["ProdTaken"]
        print("\nFeatures and target separated.")
        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)

        # --- Apply the same preprocessing as used during training ---
        # This is CRITICAL for evaluation to work correctly.
        # You need to apply the same steps (imputation, encoding, etc.) to X_test
        # as you did to the training data before training your model.
        # If you used LabelEncoding, you need to load the fitted LabelEncoders and apply them.
        # If you used imputation, you need the imputation values (median/mode from training)
        # and apply them.
        print("\nApplying preprocessing to test data...")

        # Example preprocessing (ensure this matches your training preprocessing):
        # Identify categorical columns (based on the original data types before encoding)
        categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'Designation', 'ProductPitched'] # Adjust if needed

        # Apply preprocessing for categorical and numerical features
        for col in X_test.columns:
            if col in categorical_cols:
                if X_test[col].dtype == 'object': # Check if still object type (not yet encoded)
                    # Simple imputation for demonstration (replace with actual imputation used in training)
                    # You should use the mode calculated from the TRAINING data
                    X_test[col].fillna(X_test[col].mode()[0], inplace=True) # Replace with training mode
                    # Apply encoding (replace with your actual encoding logic and fitted encoder)
                    le = LabelEncoder() # Replace with loading your fitted LabelEncoder
                    # Fit on the unique values in the test set for demonstration,
                    # but in production, fit on the training data's unique values + test set unique values
                    # or load a pre-fitted encoder.
                    all_categories = pd.concat([test_df[col], pd.Series(X_test[col].unique())]).unique() # Example to include test set unique values
                    le.fit(all_categories)
                    X_test[col] = le.transform(X_test[col])
            else: # Assume numerical
                 if X_test[col].dtype != 'object':
                    # Simple imputation for numerical features (replace with actual imputation used in training)
                    # You should use the median calculated from the TRAINING data
                    X_test[col].fillna(X_test[col].median(), inplace=True) # Replace with training median

        # Ensure the column order of X_test matches the training data
        # This is crucial for scikit-learn models. You should save and load the columns from X_train.
        # Assuming you have a list of training column names called training_feature_columns
        # X_test = X_test[training_feature_columns] # Uncomment and use your training column names


        print("Preprocessing applied to test data (ensure this matches training preprocessing).")


        # 3. Evaluate the model performance
        print("\nEvaluating model performance...")
        try:
            y_pred = loaded_model.predict(X_test)

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            print("\nModel Evaluation on Test Set:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")

            # You can save these metrics to a file or log them to an experiment tracking tool
            with open("evaluation_metrics.txt", "w") as f:
                f.write("Model Evaluation on Test Set:\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1-Score: {f1:.4f}\n")
            print("\nEvaluation metrics saved to evaluation_metrics.txt")

        except Exception as e:
            print(f"Error during model evaluation: {e}")
            import traceback
            print(traceback.format_exc())

    else:
        print("Error: 'ProdTaken' column not found in the test dataset.")
else:
    print("Skipping model evaluation: Model or test dataset not loaded successfully.")