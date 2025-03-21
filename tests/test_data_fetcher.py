import pytest
from src.data_fetcher import DataFetcher

def test_data_fetcher_initialization():
    api_key = "test_key"
    fetcher = DataFetcher(api_key)
    assert fetcher.api_key == api_key
    assert fetcher.base_url == "https://fightingtomatoes.com/API"

def test_get_sample_data():
    fetcher = DataFetcher("test_key")
    df = fetcher.get_sample_data()
    assert not df.empty
    assert 'fighter_1' in df.columns
    assert 'fighter_2' in df.columns
    assert 'performance_metric' in df.columns
    assert 'age' in df.columns

@pytest.mark.parametrize("year,event,fighter", [
    ("Any", "Any", "Any"),
    ("2023", "Any", "Any"),
    ("Any", "UFC271", "Any"),
    ("Any", "Any", "Israel Adesanya")
])
def test_fetch_fight_data(year, event, fighter):
    fetcher = DataFetcher("test_key")
    data = fetcher.fetch_fight_data(year, event, fighter)
    assert data is not None 