import os
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from tripadviser_dataset import TripAdvisorDataset


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


def create_data_loader(df, tokenizer, max_len, batch_size) -> DataLoader:
    ds = TripAdvisorDataset(
        reviews=df.review_full.to_numpy(),
        targets=df.rating_review.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=os.cpu_count())


def create_dataloaders(tokenizer: BertTokenizer,
                       max_len: int,
                       bach_size: int,
                       random_seed: int) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    df_train = read_data(Path('..') / 'data' / 'New_Delhi_reviews.csv')
    df_train['review_full'] = preprocess_data(df_train['review_full'])
    df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=random_seed)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=random_seed)
    train_dataloader = create_data_loader(df_train, tokenizer, max_len, bach_size)
    val_dataloader = create_data_loader(df_val, tokenizer, max_len, bach_size)
    test_dataloader = create_data_loader(df_test, tokenizer, max_len, bach_size)
    return train_dataloader, test_dataloader, val_dataloader, len(df_train), len(df_val)


def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples
):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in tqdm(data_loader, 'Batches: '):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)
