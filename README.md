# PCA Analysis Flask Application

This Flask web application performs Principal Component Analysis (PCA) on uploaded datasets and provides interactive visualizations of the results. The application includes logistic regression classification and various data visualization tools.

## Features

- Upload CSV datasets for PCA analysis
- Automatic data preprocessing and scaling
- Dimensionality reduction to 2 principal components
- Visualization of:
  - PCA-transformed data points
  - Training set results with decision boundaries
  - Test set results with decision boundaries
- Display of explained variance ratios
- Interactive web interface
- Real-time processing and visualization
- Comprehensive test suite with synthetic and real data

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pca-flask-app
```

2. Create and activate a virtual environment (recommended):
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
pca-flask-app/
├── app.py                  # Main Flask application
├── requirements.txt        # Project dependencies
├── README.md              # Project documentation
├── pytest.ini             # Pytest configuration
├── templates/             # HTML templates
│   └── index.html         # Main web interface
├── uploads/               # Temporary folder for file uploads
└── tests/                 # Test suite directory
    ├── __init__.py
    ├── conftest.py        # Test configurations and fixtures
    ├── test_pca.py        # Test cases
    ├── export_wine_data.py # Utility to export test data
    └── test_data/         # Test datasets directory
        ├── test_dataset.csv   # Auto-generated synthetic data
        └── wine_test_data.csv # Real wine dataset for testing
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

3. Upload your dataset:
   - The CSV file should have 13 feature columns
   - The last column (14th) should be the label/class column
   - The data should be preprocessed and cleaned
   - Missing values should be handled before upload

4. View the results:
   - The application will display the explained variance ratio
   - Three visualizations will be generated
   - Results are displayed without page reload

## Testing

### Setting Up the Test Environment

1. Test data preparation:
```bash
# Generate the wine dataset for testing
python tests/export_wine_data.py
```

2. Run the tests:
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=.

# Run specific test file
pytest tests/test_pca.py
```

### Test Suite Components

1. **Synthetic Data Tests**
   - Automatically generates test data
   - Validates PCA transformation
   - Tests classification accuracy
   - Verifies data loading and preprocessing

2. **Real Data Tests**
   - Uses scikit-learn's wine dataset
   - Validates against known good data
   - Tests end-to-end functionality
   - Verifies model performance

3. **Flask Application Tests**
   - Tests all endpoints
   - Validates file uploads
   - Verifies JSON responses
   - Tests visualization generation

### Test Configuration

The `pytest.ini` file includes settings for:
- Test discovery patterns
- Verbosity levels
- Custom markers for integration and slow tests
- Strict marker checking

## Input Data Format

The application expects CSV files with the following characteristics:
- 14 columns total
- First 13 columns contain features
- Last column contains class labels
- No missing values
- Numeric data only
- Headers are optional

Example of the first few lines of a valid CSV file:
```
feature1,feature2,...,feature13,label
1.23,4.56,...,7.89,0
2.34,5.67,...,8.90,1
```

## Visualizations

The application generates three types of plots:

1. **PCA Visualization**
   - Shows data points in the reduced 2D space
   - Different classes are color-coded
   - Helps identify clusters and patterns

2. **Training Set Results**
   - Displays decision boundaries from logistic regression
   - Shows how the model separates classes in training data
   - Includes color-coded regions for different classes

3. **Test Set Results**
   - Similar to training set visualization
   - Shows model performance on unseen data
   - Helps identify potential overfitting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Run the test suite to ensure everything passes
5. Submit a pull request

### Development Guidelines

- Write tests for all new features
- Maintain test coverage above 80%
- Follow PEP 8 style guidelines
- Update documentation as needed

## Error Handling

The application includes error handling for common issues:
- Invalid file formats
- Missing or corrupted data
- Processing errors
- Server-side exceptions

Error messages will be displayed in the web interface.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- scikit-learn for PCA and machine learning components
- Flask for the web framework
- Matplotlib for visualization capabilities
- NumPy and Pandas for data handling
- UCI Machine Learning Repository for the wine dataset

## Support

For support, please:
1. Check existing issues in the repository
2. Run tests to verify if the issue is reproducible
3. Open a new issue with test results if needed
4. Contact the maintainers

## Future Enhancements

Planned features for future releases:
- Support for different numbers of components
- Multiple classification algorithms
- More detailed statistical analysis
- Data preprocessing options
- Download capabilities for plots and results
- Batch processing of multiple files
- Custom parameter tuning
- Additional visualization options
- Support for more dataset formats