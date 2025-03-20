import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, roc_curve, auc
from imblearn.over_sampling import SMOTE
import shap
import joblib
import requests

# Fighting Tomatoes API Key
API_KEY = "a7fddea3730407eac53a9d12ac439beda9665844"


# Fetch Fight Data

def fetch_fight_data(year="Any", event="Any", fighter="Any"):
    url = f"https://fightingtomatoes.com/API/{API_KEY}/{year}/{event}/{fighter}"
    response = requests.get(url)
    if response.status_code == 200:
        print(f"Fetched data for Year: {year}, Event: {event}, Fighter: {fighter}")
        try:
            return response.json()
        except ValueError:
            print("Error: Response is not JSON formatted.")
            print("Response content:", response.text)
            return None
    else:
        print(f"Error fetching data: {response.status_code}")
        print("Response content:", response.text)
        return None


# Feature Engineering & Data Processing
def add_features(dataframe):
    dataframe['age_squared'] = dataframe['age'] ** 2
    dataframe['log_performance'] = np.log1p(dataframe['performance_metric'])
    dataframe['interaction_feature'] = dataframe['age'] * dataframe['performance_metric']
    return dataframe


def preprocess_data(dataframe, scaler_type='minmax'):
    numeric_features = ['age', 'performance_metric']
    categorical_features = ['category_name', 'gender']
    for column in categorical_features:  # Eksik sütunları varsayılan değerlerle ekle
        if column not in dataframe.columns:
            dataframe[column] = "Unknown"
    scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler, numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    return preprocessor


def analyze_balance(target_values):
    imbalance_ratio = target_values.value_counts().min() / target_values.value_counts().max()
    return imbalance_ratio < 0.5


def feature_selection(features, target):
    return SelectKBest(score_func=f_regression, k=min(5, features.shape[1])).fit_transform(features, target)


def train_ensemble_model(features, target):
    param_dist = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15], 'learning_rate': [0.01, 0.1, 0.2]}
    base_models = [
        ('RandomForest', RandomizedSearchCV(RandomForestRegressor(random_state=42), param_distributions=param_dist, n_iter=10, cv=KFold(n_splits=5), n_jobs=-1)),
        ('GradientBoosting', RandomizedSearchCV(GradientBoostingRegressor(random_state=42), param_distributions=param_dist, n_iter=10, cv=KFold(n_splits=5), n_jobs=-1))
    ]
    stacking_regressor = Pipeline(steps=[('model', StackingRegressor(estimators=base_models, final_estimator=RandomForestRegressor(random_state=42)))] )
    voting_regressor = Pipeline(steps=[('model', VotingRegressor(estimators=base_models))])
    models = {'Stacking': stacking_regressor, 'Voting': voting_regressor}
    best_models = {}
    for name, model in models.items():
        scores = cross_val_score(model, features, target, cv=KFold(n_splits=5), scoring='neg_mean_squared_error')
        print(f"{name} Cross-Validation MSE: {-np.mean(scores)}")
        model.fit(features, target)
        best_models[name] = model
    joblib.dump(best_models, 'best_models.pkl')
    return best_models


def evaluate_models(models, features, target):
    for name, model in models.items():
        target_pred = model.predict(features)
        print(f"--- {name} Model ---")
        print(f"MSE: {mean_squared_error(target, target_pred)}")
        print(f"MAE: {mean_absolute_error(target, target_pred)}")
        print(f"MAPE: {mean_absolute_percentage_error(target, target_pred)}")
        explainer = shap.Explainer(model.predict, features)
        shap_values = explainer(features)
        shap.summary_plot(shap_values, features)
        plt.figure(figsize=(10, 6))
        sns.residplot(x=target, y=target_pred, lowess=True)
        plt.xlabel("Actual Values")
        plt.ylabel("Residuals")
        plt.title(f"Residual Plot - {name}")
        plt.show()


def predict_fight_winners(models, df):
    if 'Stacking' in models:
        model = models['Stacking']  # En iyi modeli seçtik
        fights = []
        for _, row in df.iterrows():
            fighter_1 = row['fighter_1']
            fighter_2 = row['fighter_2']
            input_data = pd.DataFrame([{**row}])
            input_data = add_features(input_data)
            preprocessor = preprocess_data(input_data, scaler_type='standard')
            input_data_preprocessed = preprocessor.fit_transform(input_data)
            winner = fighter_1 if model.predict(input_data_preprocessed)[0] > 0.5 else fighter_2
            fights.append({
                'Event': row['event'],
                'Fighter 1': fighter_1,
                'Fighter 2': fighter_2,
                'Predicted Winner': winner
            })
        predictions = pd.DataFrame(fights)
        predictions.to_csv('fight_predictions.csv', index=False)
        print(predictions)
        return predictions
    else:
        print("No valid model found for predictions.")
        return None


# Main Execution


fight_data = fetch_fight_data(year="Any", event="Any", fighter="Any")

if fight_data:
    df = pd.DataFrame(fight_data)
    df['performance_metric'] = df.get('fighting_tomatoes_aggregate_rating', np.random.randint(50, 100))
    df['age'] = 2024 - 1987  # Sahte yaş
else:
    print("Using placeholder data due to API issues.")
    fight_data = [
        {
            "date": "2022-05-22",
            "promotion": "UFC",
            "event": "271",
            "main_or_prelim": "Main",
            "card_placement": 1,
            "fighter_1": "Israel Adesanya",
            "fighter_2": "Robert Whittaker",
            "rematch": 2,
            "winner": "Israel Adesanya",
            "method": "Decision",
            "round": 5,
            "time": "5:00",
            "fighting_tomatoes_aggregate_rating": 62,
        }
    ]
    df = pd.DataFrame(fight_data)
    df['performance_metric'] = df['fighting_tomatoes_aggregate_rating']
    df['age'] = 2024 - 1987  # Sahte yaş

df = add_features(df)
preprocessor = preprocess_data(df, scaler_type='standard')
X = preprocessor.fit_transform(df)
y = df['performance_metric']

if analyze_balance(pd.Series(y)):
    X, y = SMOTE().fit_resample(X, y)

X_selected = feature_selection(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
models = train_ensemble_model(X_train, y_train)
evaluate_models(models, X_test, y_test)

# Predict fight winners
predicted_winners = predict_fight_winners(models, df)


# EDA - Veri Görselleştirme

def plot_correlation(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.show()


def plot_distribution(df, column):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=20, kde=True)
    plt.title(f'{column} Distribution')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


def plot_predictions(y_test, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted Values - {model_name}')
    plt.show()


def plot_roc_curve(y_test, y_pred):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


def plot_error_distribution(y_test, y_pred, model_name):
    errors = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=20, kde=True)
    plt.title(f'Error Distribution - {model_name}')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.show()