from datasets import load_dataset
from huggingface_hub import notebook_login, HfApi, create_repo, upload_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib
import os

# Login to Hugging Face (you will be prompted to enter your token)
# notebook_login() # This should be handled outside the script for automation

# Define your dataset repository ID on Hugging Face
dataset_repo_id = "LalithaShiva/tourism_prediction_dataset" # Replace with your actual dataset repository ID
model_repo_id = "LalithaShiva/tourism_prediction_model" # Replace with your desired model repository ID

# 1. Load the train and test data from the Hugging Face data space
try:
    dataset = load_dataset(dataset_repo_id, split=['train', 'test'])
    train_dataset = dataset[0]
    test_dataset = dataset[1]
    print("Train and test datasets loaded successfully!")
except Exception as e:
    print(f"Error loading datasets: {e}")
    train_dataset = None
    test_dataset = None

if train_dataset and test_dataset:
    # Convert datasets to pandas DataFrames for easier handling with scikit-learn
    train_df = train_dataset.to_pandas()
    test_df = test_dataset.to_pandas()

    # Separate features (X) and target (y)
    X_train = train_df.drop("ProdTaken", axis=1)
    y_train = train_df["ProdTaken"]
    X_test = test_df.drop("ProdTaken", axis=1)
    y_test = test_df["ProdTaken"]

    # Handle categorical features and potential missing values - IMPORTANT: Match preprocessing from data_preparation
    # This is a basic example, ensure it aligns with what you did in data_preparation.py
    # You might need to save and load your LabelEncoders and imputation values.
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            # Simple imputation for demonstration
            X_train[col].fillna(X_train[col].mode()[0], inplace=True)
            X_test[col].fillna(X_test[col].mode()[0], inplace=True)

            # Encode categorical features
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            X_test[col] = le.transform(X_test[col])
        else:
            # Simple imputation for numerical features
            X_train[col].fillna(X_train[col].median(), inplace=True)
            X_test[col].fillna(X_test[col].median(), inplace=True)


    # 2. Define a model and parameters
    model = RandomForestClassifier(random_state=42)
    parameters = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30]
    }
    print("\nRandomForestClassifier model and parameters defined.")

    # 3. Tune the model with the defined parameters
    grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=3, n_jobs=-1, verbose=2)
    print("\nStarting grid search...")
    grid_search.fit(X_train, y_train)
    best_parameters = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_
    print("Grid Search completed.")
    print("Best parameters found:", best_parameters)
    print("Best cross-validation score:", best_score)

    # 4. Log all the tuned parameters
    print("\nTuned Parameters:")
    for param, value in best_parameters.items():
        print(f"- {param}: {value}")
    # Example of logging to a file
    with open("tuned_parameters.txt", "w") as f:
        f.write("Tuned Parameters:\n")
        for param, value in best_parameters.items():
            f.write(f"- {param}: {value}\n")
    print("Tuned parameters logged to tuned_parameters.txt")


    # 5. Evaluate the model performance
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("\nModel Evaluation on Test Set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


    # 6. Register the best model in the Hugging Face model hub
    model_save_path = "best_random_forest_model.joblib"
    joblib.dump(best_model, model_save_path)
    print(f"\nBest model saved locally to {model_save_path}")

    api = HfApi()
    try:
        create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True) # Use exist_ok=True
        print(f"Model repository '{model_repo_id}' created or already exists.")
    except Exception as e:
        print(f"Could not create model repository: {e}")

    try:
        upload_file(
            path_or_fileobj=model_save_path,
            path_in_repo="best_model.joblib",
            repo_id=model_repo_id,
            repo_type="model",
        )
        print(f"Best model uploaded to Hugging Face Hub: {model_repo_id}")
    except Exception as e:
        print(f"Error uploading model to Hugging Face: {e}")

    # Upload tuned_parameters.txt as well
    try:
        upload_file(
            path_or_fileobj="tuned_parameters.txt",
            path_in_repo="tuned_parameters.txt",
            repo_id=model_repo_id,
            repo_type="model",
        )
        print(f"Tuned parameters file uploaded to Hugging Face Hub: {model_repo_id}")
    except Exception as e:
        print(f"Error uploading tuned parameters file to Hugging Face: {e}")

else:
    print("Skipping model building and tracking: Datasets were not loaded successfully.")