import pytest
import numpy as np
import pandas as pd
from src.models import ModelTrainer

@pytest.fixture
def sample_data():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    return X, y

@pytest.fixture
def sample_fight_data():
    return pd.DataFrame({
        'event': ['UFC271', 'UFC272', 'UFC273'],
        'fighter_1': ['Fighter1', 'Fighter2', 'Fighter3'],
        'fighter_2': ['Fighter4', 'Fighter5', 'Fighter6'],
        'performance_metric': [80, 85, 90],
        'age': [25, 30, 35]
    })

def test_model_trainer_initialization():
    trainer = ModelTrainer()
    assert trainer.random_state == 42
    assert isinstance(trainer.param_dist, dict)
    assert 'n_estimators' in trainer.param_dist

def test_analyze_balance():
    trainer = ModelTrainer()
    balanced_data = pd.Series([1, 1, 1, 2, 2, 2])
    imbalanced_data = pd.Series([1, 1, 1, 1, 1, 1, 2, 2])
    
    assert not trainer.analyze_balance(balanced_data)
    assert trainer.analyze_balance(imbalanced_data)

def test_train_ensemble_model(sample_data):
    trainer = ModelTrainer()
    X, y = sample_data
    models = trainer.train_ensemble_model(X, y)
    
    assert isinstance(models, dict)
    assert 'Stacking' in models
    assert 'Voting' in models
    assert hasattr(models['Stacking'], 'predict')
    assert hasattr(models['Voting'], 'predict')

def test_evaluate_models(sample_data):
    trainer = ModelTrainer()
    X, y = sample_data
    models = trainer.train_ensemble_model(X, y)
    trainer.evaluate_models(models, X, y)
    # No assertions needed as this is a visualization function

def test_predict_fight_winners(sample_data, sample_fight_data):
    trainer = ModelTrainer()
    X, y = sample_data
    models = trainer.train_ensemble_model(X, y)
    predictions = trainer.predict_fight_winners(models, sample_fight_data)
    
    assert isinstance(predictions, pd.DataFrame)
    assert 'Event' in predictions.columns
    assert 'Fighter 1' in predictions.columns
    assert 'Fighter 2' in predictions.columns
    assert 'Predicted Winner' in predictions.columns 