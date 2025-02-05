# tests/export_wine_data.py
import os
import pandas as pd
from sklearn.datasets import load_wine


def export_wine_dataset():
    # Load the wine dataset
    wine = load_wine()

    # Create DataFrame with feature names
    df = pd.DataFrame(wine.data, columns=wine.feature_names)

    # Add target column
    df['Label'] = wine.target

    # Create test_data directory if it doesn't exist
    test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    os.makedirs(test_data_dir, exist_ok=True)

    # Export to CSV
    output_path = os.path.join(test_data_dir, 'wine_test_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Dataset exported to: {output_path}")
    print(f"Shape of dataset: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())

    return output_path


if __name__ == "__main__":
    export_wine_dataset()
