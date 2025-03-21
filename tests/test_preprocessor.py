import pytest
import pandas as pd
import numpy as np
from src.preprocessor import DataPreprocessor

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'age': [25, 30, 35],
        'performance_metric': [80, 85, 90],
        'category_name': ['Lightweight', 'Welterweight', 'Middleweight'],
        'gender': ['Male', 'Male', 'Male']
    })

def test_preprocessor_initialization():
    preprocessor = DataPreprocessor()
    assert preprocessor.scaler_type == 'minmax'
    assert preprocessor.preprocessor is None
    assert preprocessor.feature_selector is None

def test_add_features(sample_data):
    preprocessor = DataPreprocessor()
    df = preprocessor.add_features(sample_data)
    assert 'age_squared' in df.columns
    assert 'log_performance' in df.columns
    assert 'interaction_feature' in df.columns

def test_prepare_preprocessor():
    preprocessor = DataPreprocessor()
    transformer = preprocessor.prepare_preprocessor()
    assert transformer is not None
    assert len(transformer.transformers) == 2

def test_preprocess_data(sample_data):
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_data(sample_data)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, pd.Series)
    assert len(X) == len(sample_data)
    assert len(y) == len(sample_data)

def test_select_features(sample_data):
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess_data(sample_data)
    X_selected = preprocessor.select_features(X, y, k=3)
    assert isinstance(X_selected, np.ndarray)
    assert X_selected.shape[1] <= 3 