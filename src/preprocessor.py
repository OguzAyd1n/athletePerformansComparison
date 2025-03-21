import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from typing import Tuple, List

class DataPreprocessor:
    def __init__(self, scaler_type: str = 'minmax'):
        self.scaler_type = scaler_type
        self.preprocessor = None
        self.feature_selector = None
        self.numeric_features = ['age', 'performance_metric']
        self.categorical_features = ['category_name', 'gender']

    def add_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features to the dataset.
        
        Args:
            dataframe: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df = dataframe.copy()
        df['age_squared'] = df['age'] ** 2
        df['log_performance'] = np.log1p(df['performance_metric'])
        df['interaction_feature'] = df['age'] * df['performance_metric']
        return df

    def prepare_preprocessor(self) -> ColumnTransformer:
        """
        Prepare the preprocessor with specified transformations.
        
        Returns:
            ColumnTransformer object
        """
        scaler = MinMaxScaler() if self.scaler_type == 'minmax' else StandardScaler()
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', scaler, self.numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ]
        )
        return self.preprocessor

    def select_features(self, features: np.ndarray, target: np.ndarray, k: int = 5) -> np.ndarray:
        """
        Select the best k features using f_regression.
        
        Args:
            features: Feature matrix
            target: Target variable
            k: Number of features to select
            
        Returns:
            Selected feature matrix
        """
        self.feature_selector = SelectKBest(score_func=f_regression, k=min(k, features.shape[1]))
        return self.feature_selector.fit_transform(features, target)

    def preprocess_data(self, dataframe: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the input data.
        
        Args:
            dataframe: Input DataFrame
            
        Returns:
            Tuple of (features, target)
        """
        # Add missing columns if they don't exist
        for column in self.categorical_features:
            if column not in dataframe.columns:
                dataframe[column] = "Unknown"

        # Add engineered features
        df = self.add_features(dataframe)
        
        # Prepare and fit preprocessor
        preprocessor = self.prepare_preprocessor()
        X = preprocessor.fit_transform(df)
        
        # Get target variable
        y = df['performance_metric']
        
        return X, y 