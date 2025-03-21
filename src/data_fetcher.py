import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

class DataFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://fightingtomatoes.com/API"

    def fetch_fight_data(self, year: str = "Any", event: str = "Any", fighter: str = "Any") -> Optional[Dict[str, Any]]:
        """
        Fetch fight data from the Fighting Tomatoes API.
        
        Args:
            year: Year of the fight
            event: Event name
            fighter: Fighter name
            
        Returns:
            Dictionary containing fight data or None if request fails
        """
        url = f"{self.base_url}/{self.api_key}/{year}/{event}/{fighter}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None

    def get_sample_data(self) -> pd.DataFrame:
        """
        Get sample data for testing purposes.
        
        Returns:
            DataFrame containing sample fight data
        """
        sample_data = [
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
        df = pd.DataFrame(sample_data)
        df['performance_metric'] = df['fighting_tomatoes_aggregate_rating']
        df['age'] = 2024 - 1987  # Placeholder age
        return df 