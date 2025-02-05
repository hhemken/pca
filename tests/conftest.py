# tests/conftest.py
import pytest
import os
import pandas as pd
import numpy as np


@pytest.fixture
def test_data_path():
    """Returns the path to the test data directory"""
    return os.path.join(os.path.dirname(__file__), 'test_data')


@pytest.fixture
def sample_dataset():
    """Creates a sample dataset for testing"""
    np.random.seed(42)
    n_samples = 100
    n_features = 13

    # Generate random features
    X = np.random.randn(n_samples, n_features)

    # Generate labels (3 classes)
    y = np.random.randint(0, 3, n_samples)

    # Create DataFrame
    columns = [f'feature_{i + 1}' for i in range(n_features)]
    columns.append('label')

    data = np.column_stack([X, y])
    df = pd.DataFrame(data, columns=columns)

    return df


@pytest.fixture
def test_csv_path(test_data_path, sample_dataset):
    """Creates and saves a test CSV file"""
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)

    file_path = os.path.join(test_data_path, 'test_dataset.csv')
    sample_dataset.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def app():
    """Creates a test Flask application"""
    from app import app as flask_app
    flask_app.config['TESTING'] = True
    return flask_app
