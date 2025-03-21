import os
from dotenv import load_dotenv
from src.data_fetcher import DataFetcher
from src.preprocessor import DataPreprocessor
from src.models import ModelTrainer
from src.visualization import Visualizer
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('FIGHTING_TOMATOES_API_KEY', 'a7fddea3730407eac53a9d12ac439beda9665844')

    # Initialize components
    data_fetcher = DataFetcher(api_key)
    preprocessor = DataPreprocessor()
    model_trainer = ModelTrainer()
    visualizer = Visualizer()

    # Fetch data
    fight_data = data_fetcher.fetch_fight_data()
    if not fight_data:
        print("Using sample data due to API issues.")
        df = data_fetcher.get_sample_data()
    else:
        df = pd.DataFrame(fight_data)

    # Preprocess data
    X, y = preprocessor.preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    models = model_trainer.train_ensemble_model(X_train, y_train)

    # Evaluate models
    model_trainer.evaluate_models(models, X_test, y_test)

    # Make predictions
    predictions = model_trainer.predict_fight_winners(models, df)
    if predictions is not None:
        print("\nPredictions:")
        print(predictions)

    # Visualizations
    visualizer.plot_correlation(df)
    visualizer.plot_distribution(df, 'performance_metric')
    visualizer.plot_predictions(y_test, models['Stacking'].predict(X_test), 'Stacking Model')
    visualizer.plot_error_distribution(y_test, models['Stacking'].predict(X_test), 'Stacking Model')

if __name__ == "__main__":
    main() 