from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class TripAdvisorDataset(Dataset):
    """
    Torch dataset for load data from kaggle competition

    :param reviews: reviews data as pd.Series
    :param targets: targets data as pd.Series
    :param tokenizer: tokenizer from transformers library
    :param max_len: maximum length of review text"""

    def __init__(self, reviews: pd.Series, targets: pd.Series, tokenizer: BertTokenizer, max_len: int):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item) -> Dict[str, torch.Tensor]:
        """
        Returns dictionary with keys:
        'review_text': review text as string
        'input_ids': tokenized review text
        'attention_mask': attention mask for BERT model
        'targets': target for review text
        """
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }