from datasets import load_dataset, DatasetDict
from huggingface_hub import notebook_login
import pandas as pd # Import pandas for potential data manipulation if needed


# Login to Hugging Face (you will be prompted to enter your token)
notebook_login()

# Define your dataset repository ID on Hugging Face
dataset_repo_id = "LalithaShiva/tourism_prediction_dataset" # Replace with your actual dataset repository ID

# 1. Load the dataset directly from the Hugging Face data space.
try:
    dataset = load_dataset(dataset_repo_id)
    print("Dataset loaded successfully!")
    print(dataset)
except Exception as e:
    print(f"Error loading dataset: {e}")
    dataset = None

if dataset:
    # 2. Perform data cleaning and remove any unnecessary columns.
    # Inspect the dataset to identify columns to remove (e.g., 'Unnamed: 0', 'CustomerID')
    # You may need to adjust the list of columns to drop based on your analysis
    columns_to_drop = ['Unnamed: 0', 'CustomerID'] # Adjust this list as needed

    # Check if the columns exist before trying to remove them
    columns_exist = all(col in dataset['train'].features for col in columns_to_drop)

    if columns_exist:
        try:
            # Apply removal to each split in the DatasetDict
            dataset = dataset.remove_columns(columns_to_drop)
            print(f"Columns {columns_to_drop} removed successfully!")
            print(dataset)
        except Exception as e:
            print(f"Error removing columns: {e}")
    else:
        print(f"Skipping column removal: Some columns in {columns_to_drop} not found in the dataset.")


    # You might want to perform other cleaning steps here, such as handling missing values,
    # converting data types, or encoding categorical features.
    # Example: Handling missing values (simple imputation)
    # for split in dataset.keys():
    #     for col in dataset[split].features:
    #         if dataset[split].features[col].dtype == 'float64' or dataset[split].features[col].dtype == 'int64':
    #             # Simple median imputation for numerical columns
    #             median_value = pd.DataFrame(dataset[split]).fillna(0).median()[col] # Calculate median safely
    #             dataset[split] = dataset[split].map(lambda x: {col: x[col] if x[col] is not None else median_value})
    #         elif dataset[split].features[col].dtype == 'string':
    #             # Simple mode imputation for string columns
    #             mode_value = pd.DataFrame(dataset[split]).fillna('').mode().iloc[0][col] # Calculate mode safely
    #             dataset[split] = dataset[split].map(lambda x: {col: x[col] if x[col] != '' else mode_value})


    # 3. Split the cleaned dataset into training and testing sets, and save them locally.
    # Assuming the dataset is already a DatasetDict with 'train' split
    if 'train' in dataset:
        try:
            # Perform train-test split on the 'train' split
            train_test_split = dataset['train'].train_test_split(test_size=0.2, seed=42) # Added seed for reproducibility
            dataset_dict_local = DatasetDict({
                'train': train_test_split['train'],
                'test': train_test_split['test']
            })
            print("Dataset split into training and testing sets successfully!")
            print(dataset_dict_local)

            # Save the training and testing sets locally
            # Ensure the local directory exists
            os.makedirs("tourism_prediction/data", exist_ok=True)
            dataset_dict_local['train'].to_csv("tourism_prediction/data/train_dataset.csv")
            dataset_dict_local['test'].to_csv("tourism_prediction/data/test_dataset.csv")
            print("Training and testing datasets saved locally!")

        except Exception as e:
            print(f"Error splitting or saving dataset: {e}")
            dataset_dict_local = None

        # 4. Upload the resulting train and test datasets back to the Hugging Face data space.
        if dataset_dict_local:
            try:
                # Push the new DatasetDict to the Hugging Face Hub
                # This will overwrite the existing dataset splits or add new ones
                dataset_dict_local.push_to_hub(dataset_repo_id)
                print("Training and testing datasets uploaded to Hugging Face!")
            except Exception as e:
                print(f"Error uploading datasets to Hugging Face: {e}")
    else:
        print("Skipping train-test split and upload: 'train' split not found in the loaded dataset.")