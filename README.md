# Athlete Performance Comparison

This project is a machine learning-based system for analyzing and predicting athlete performance in combat sports. It uses various machine learning models to analyze historical fight data and predict potential outcomes.

## Features

- Data fetching from Fighting Tomatoes API
- Feature engineering and preprocessing
- Ensemble machine learning models (Random Forest, Gradient Boosting)
- Performance evaluation metrics
- Visualization tools for data analysis
- Fight winner prediction system

## Installation

1. Clone the repository:
```bash
git clone https://github.com/OguzAyd1n/athletePerformansComparison.git
cd athletePerformansComparison
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Set up your API key in the configuration file
2. Run the main script:
```bash
python orn.py
```

## Project Structure

- `orn.py`: Main script containing the core functionality
- `dns.py`: DNS-related utilities
- `competitor_profiles.csv`: Athlete profile data
- `match_data.csv`: Historical match data
- `competitions.csv`: Competition information

## Data Sources

The project uses data from:
- Fighting Tomatoes API
- Historical fight records
- Athlete profiles

## Models Used

- Random Forest Regressor
- Gradient Boosting Regressor
- Stacking Ensemble
- Voting Ensemble

## Performance Metrics

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- ROC Curve Analysis

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Fighting Tomatoes API for providing fight data
- Contributors and maintainers of the machine learning libraries used 