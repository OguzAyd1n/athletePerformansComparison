import pytest
import pandas as pd
import numpy as np
from src.visualization import Visualizer

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'age': [25, 30, 35],
        'performance_metric': [80, 85, 90],
        'age_squared': [625, 900, 1225],
        'log_performance': [4.38, 4.44, 4.50],
        'interaction_feature': [2000, 2550, 3150]
    })

@pytest.fixture
def sample_predictions():
    y_test = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
    return y_test, y_pred

def test_plot_correlation(sample_data):
    visualizer = Visualizer()
    visualizer.plot_correlation(sample_data)
    # No assertions needed as this is a visualization function

def test_plot_distribution(sample_data):
    visualizer = Visualizer()
    visualizer.plot_distribution(sample_data, 'performance_metric')
    # No assertions needed as this is a visualization function

def test_plot_predictions(sample_predictions):
    visualizer = Visualizer()
    y_test, y_pred = sample_predictions
    visualizer.plot_predictions(y_test, y_pred, 'Test Model')
    # No assertions needed as this is a visualization function

def test_plot_roc_curve(sample_predictions):
    visualizer = Visualizer()
    y_test, y_pred = sample_predictions
    visualizer.plot_roc_curve(y_test, y_pred)
    # No assertions needed as this is a visualization function

def test_plot_error_distribution(sample_predictions):
    visualizer = Visualizer()
    y_test, y_pred = sample_predictions
    visualizer.plot_error_distribution(y_test, y_pred, 'Test Model')
    # No assertions needed as this is a visualization function 