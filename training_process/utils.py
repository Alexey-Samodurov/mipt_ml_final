from pathlib import Path
from typing import Union, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def read_data(csv_path: Union[Path, str]) -> pd.DataFrame:
    """Reads data from csv file and returns it as pandas DataFrame
    :param csv_path: path to csv file
    :return: pandas DataFrame
    """
    return pd.read_csv(csv_path, dtype={'rating_review': int, 'review_full': str}).dropna()


def preprocess_data(review: pd.Series, max_len: int = 400) -> pd.Series:
    """
    Clean data and returns new pd.Series

    :param max_len: maximum length of review text
    :param review: series with rewie data needs to be cleaned
    """
    patterns = [['&#039;', ''], [r'[^\w\d\s]', ' '], [r'[^\x00-\x7F]+', ' '],
                [r'^\s+|\s+?$', ''], [r'\s+', ' '], [r'\.{2,}', ' ']]
    review = review.str.lower()
    for patt in tqdm(patterns, desc='Cleaning data'):
        review = review.str.replace(*patt, regex=True)
    review = review.apply(lambda x: ' '.join(x.split()[:max_len]) if len(x.split()) > max_len else x)
    review = review.apply(lambda x: ' '.join(x.split() if len(x.split()) > 1 else x))
    return review


def train_test_split_data(full_data: pd.DataFrame,
                          test_size: float = 0.2,
                          random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data into train and test sets by given test_size and random_state
    :param full_data: pandas DataFrame with data
    :param test_size: float, default=0.2
    :param random_state: int, default=42
    :return: tuple of pandas DataFrames"""
    train_df, test_df = train_test_split(full_data, test_size=test_size, random_state=random_state)
    print(f'Train data shape: {train_df.shape}\nTest data shape: {test_df.shape}')
    return train_df, test_df


if __name__ == '__main__':
    data = read_data(r'..\data\New_Delhi_reviews.csv')
    data['review_full'] = preprocess_data(data['review_full'])
    print(data.head())
