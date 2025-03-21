import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from imblearn.over_sampling import SMOTE
import joblib
import shap
from typing import Dict, Any, Tuple

class ModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.param_dist = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'learning_rate': [0.01, 0.1, 0.2]
        }

    def analyze_balance(self, target_values: pd.Series) -> bool:
        """
        Analyze if the target variable is imbalanced.
        
        Args:
            target_values: Target variable values
            
        Returns:
            Boolean indicating if data is imbalanced
        """
        imbalance_ratio = target_values.value_counts().min() / target_values.value_counts().max()
        return imbalance_ratio < 0.5

    def train_ensemble_model(self, features: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
        """
        Train ensemble models using stacking and voting.
        
        Args:
            features: Feature matrix
            target: Target variable
            
        Returns:
            Dictionary of trained models
        """
        base_models = [
            ('RandomForest', RandomizedSearchCV(
                RandomForestRegressor(random_state=self.random_state),
                param_distributions=self.param_dist,
                n_iter=10,
                cv=KFold(n_splits=5),
                n_jobs=-1
            )),
            ('GradientBoosting', RandomizedSearchCV(
                GradientBoostingRegressor(random_state=self.random_state),
                param_distributions=self.param_dist,
                n_iter=10,
                cv=KFold(n_splits=5),
                n_jobs=-1
            ))
        ]

        stacking_regressor = Pipeline(steps=[
            ('model', StackingRegressor(
                estimators=base_models,
                final_estimator=RandomForestRegressor(random_state=self.random_state)
            ))
        ])

        voting_regressor = Pipeline(steps=[
            ('model', VotingRegressor(estimators=base_models))
        ])

        models = {'Stacking': stacking_regressor, 'Voting': voting_regressor}
        best_models = {}

        for name, model in models.items():
            scores = cross_val_score(
                model,
                features,
                target,
                cv=KFold(n_splits=5),
                scoring='neg_mean_squared_error'
            )
            print(f"{name} Cross-Validation MSE: {-np.mean(scores)}")
            model.fit(features, target)
            best_models[name] = model

        joblib.dump(best_models, 'best_models.pkl')
        return best_models

    def evaluate_models(self, models: Dict[str, Any], features: np.ndarray, target: np.ndarray) -> None:
        """
        Evaluate trained models using various metrics.
        
        Args:
            models: Dictionary of trained models
            features: Feature matrix
            target: Target variable
        """
        for name, model in models.items():
            target_pred = model.predict(features)
            print(f"--- {name} Model ---")
            print(f"MSE: {mean_squared_error(target, target_pred)}")
            print(f"MAE: {mean_absolute_error(target, target_pred)}")
            print(f"MAPE: {mean_absolute_percentage_error(target, target_pred)}")

            # SHAP analysis
            explainer = shap.Explainer(model.predict, features)
            shap_values = explainer(features)
            shap.summary_plot(shap_values, features)

    def predict_fight_winners(self, models: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict winners for fights using the best model.
        
        Args:
            models: Dictionary of trained models
            df: DataFrame containing fight data
            
        Returns:
            DataFrame with predictions
        """
        if 'Stacking' not in models:
            print("No valid model found for predictions.")
            return None

        model = models['Stacking']
        fights = []

        for _, row in df.iterrows():
            fighter_1 = row['fighter_1']
            fighter_2 = row['fighter_2']
            input_data = pd.DataFrame([{**row}])
            
            # Preprocess input data
            preprocessor = DataPreprocessor()
            input_data_preprocessed, _ = preprocessor.preprocess_data(input_data)
            
            winner = fighter_1 if model.predict(input_data_preprocessed)[0] > 0.5 else fighter_2
            fights.append({
                'Event': row['event'],
                'Fighter 1': fighter_1,
                'Fighter 2': fighter_2,
                'Predicted Winner': winner
            })

        predictions = pd.DataFrame(fights)
        predictions.to_csv('fight_predictions.csv', index=False)
        return predictions 