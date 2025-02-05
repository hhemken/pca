# tests/test_pca.py
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os


def test_data_loading(test_csv_path):
    """Test if the data can be loaded correctly"""
    assert os.path.exists(test_csv_path), "Test CSV file does not exist"

    df = pd.read_csv(test_csv_path)
    assert df.shape[1] == 14, "Dataset should have 14 columns (13 features + 1 label)"
    assert not df.isnull().any().any(), "Dataset should not contain any null values"


def test_pca_transformation(sample_dataset):
    """Test the PCA transformation process"""
    # Prepare data
    X = sample_dataset.iloc[:, 0:13].values
    y = sample_dataset.iloc[:, 13].values

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Scale features
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    # Apply PCA
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Tests
    assert X_train_pca.shape[1] == 2, "PCA should reduce to 2 components"
    assert pca.explained_variance_ratio_.shape[0] == 2, "Should have 2 explained variance ratios"
    assert np.all(pca.explained_variance_ratio_ >= 0), "Explained variance ratios should be non-negative"
    assert np.sum(pca.explained_variance_ratio_) <= 1, "Sum of explained variance ratios should not exceed 1"


def test_logistic_regression(sample_dataset):
    """Test the logistic regression classification"""
    # Prepare data
    X = sample_dataset.iloc[:, 0:13].values
    y = sample_dataset.iloc[:, 13].values

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Scale features
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    # Apply PCA
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Fit logistic regression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train_pca, y_train)

    # Make predictions
    y_pred = classifier.predict(X_test_pca)

    # Tests
    assert len(y_pred) == len(y_test), "Prediction length should match test set length"
    assert len(np.unique(y_pred)) <= len(
        np.unique(y_test)), "Number of predicted classes should not exceed actual classes"
    assert classifier.classes_.shape[0] == len(np.unique(y)), "Classifier should recognize all classes"


def test_flask_app(app, test_csv_path):
    """Test the Flask application endpoints"""
    with app.test_client() as client:
        # Test home page
        response = client.get('/')
        assert response.status_code == 200

        # Test PCA analysis endpoint
        with open(test_csv_path, 'rb') as f:
            response = client.post(
                '/perform_pca',
                data={'file': (f, 'test_dataset.csv')},
                content_type='multipart/form-data'
            )

        assert response.status_code == 200
        data = response.get_json()

        assert 'success' in data
        assert 'explained_variance' in data
        assert 'pca_plot' in data
        assert 'train_plot' in data
        assert 'test_plot' in data

        assert len(data['explained_variance']) == 2
        assert all(isinstance(v, float) for v in data['explained_variance'])
