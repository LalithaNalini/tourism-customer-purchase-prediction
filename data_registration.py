from huggingface_hub import notebook_login
from datasets import Dataset

# Login to Hugging Face (you will be prompted to enter your token)
notebook_login()

# Replace 'path/to/your/data.csv' with the actual path to your data file
# You might need to adjust the loading method based on your data format (e.g., json, parquet)
data_file_path = "tourism_prediction/data/tourism.csv" # Replace with your actual file path

try:
    dataset = Dataset.from_csv(data_file_path)
    print(f"Dataset loaded successfully from {data_file_path}")

    # Push the dataset to the Hugging Face Hub
    # Replace 'your-username/your-dataset-name' with your desired repository name on Hugging Face
    dataset.push_to_hub("LalithaShiva/tourism_prediction_dataset")
    print("Dataset registered on Hugging Face!")

except FileNotFoundError:
    print(f"Error: Data file not found at {data_file_path}. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")