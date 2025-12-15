import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

RAW_DATA_PATH = Path("data/raw_data.csv")
PROCESSED_DATA_PATH = Path("data/processed_data.csv")


def preprocess_data():
    # Load raw data
    df = pd.read_csv(RAW_DATA_PATH)

    # Drop rows with missing values
    df = df.dropna()

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=["object"]).columns

    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    # Save processed data
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Data preprocessing completed successfully")


if __name__ == "__main__":
    preprocess_data()

