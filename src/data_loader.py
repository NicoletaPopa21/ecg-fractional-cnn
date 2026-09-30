import pandas as pd
import kagglehub
from tensorflow.keras.utils import to_categorical

def load_mitbih_data():
    """
    Downloads and loads the MIT-BIH Arrhythmia dataset from Kaggle.
    
    Returns:
        X_train_raw, y_train, X_test_raw, y_test: Prepared numpy arrays.
    """
    print("Downloading dataset via kagglehub...")
    path = kagglehub.dataset_download("shayanfazeli/heartbeat")
    
    print("Loading CSV files into memory...")
    train_df = pd.read_csv(f"{path}/mitbih_train.csv", header=None)
    test_df = pd.read_csv(f"{path}/mitbih_test.csv", header=None)
    
    # Extract features and labels
    X_train_raw = train_df.iloc[:, :-1].values
    y_train = to_categorical(train_df.iloc[:, -1].values)
    
    X_test_raw = test_df.iloc[:, :-1].values
    y_test = to_categorical(test_df.iloc[:, -1].values)
    
    print("Data loaded successfully.")
    return X_train_raw, y_train, X_test_raw, y_test
