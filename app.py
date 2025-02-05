# app.py
from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.colors import ListedColormap
import os

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def create_pca_plot(X_train, y_train):
    plt.figure(figsize=(10, 6))
    colors = ["r", "g", "b"]
    labels = ["Class 1", "Class 2", "Class 3"]
    for i, color, label in zip(np.unique(y_train), colors, labels):
        plt.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1],
                    color=color, label=label)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()

    # Save plot to memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Encode plot as base64 string
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    return plot_url


def create_prediction_plot(X_set, y_set, classifier, title):
    plt.figure(figsize=(10, 6))
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                                   stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1,
                                   stop=X_set[:, 1].max() + 1, step=0.01))

    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
                                                      X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('yellow', 'white', 'aquamarine')))

    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    color=ListedColormap(('red', 'green', 'blue'))(i), label=j)

    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()

    # Save plot to memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Encode plot as base64 string
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    return plot_url


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/perform_pca', methods=['POST'])
def perform_pca():
    try:
        # Get uploaded file
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded'})

        # Save file temporarily
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Read dataset
        dataset = pd.read_csv(filepath)

        # Remove temporary file
        os.remove(filepath)

        # Prepare data
        X = dataset.iloc[:, 0:13].values
        y = dataset.iloc[:, 13].values

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Scale features
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Apply PCA
        pca = PCA(n_components=2)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        # Get explained variance
        explained_variance = pca.explained_variance_ratio_

        # Fit logistic regression
        classifier = LogisticRegression(random_state=0)
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = classifier.predict(X_test)

        # Create plots
        pca_plot = create_pca_plot(X_train, y_train)
        train_plot = create_prediction_plot(X_train, y_train, classifier,
                                            'Logistic Regression (Training set)')
        test_plot = create_prediction_plot(X_test, y_test, classifier,
                                           'Logistic Regression (Test set)')

        return jsonify({
            'success': True,
            'explained_variance': explained_variance.tolist(),
            'pca_plot': pca_plot,
            'train_plot': train_plot,
            'test_plot': test_plot
        })

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
