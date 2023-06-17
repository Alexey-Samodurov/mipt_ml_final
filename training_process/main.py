import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import BertTokenizer
import pytorch_lightning as pl
from trapadvisor_data_loader import TripAdvisorDataModule, TripAdvisorClassifier
from utils import train_test_split_data, preprocess_data, read_data

BERT_MODEL_NAME = 'bert-base-cased'

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, return_dict=True)

data = read_data('../data/New_Delhi_reviews.csv')
data['review_full'] = preprocess_data(data['review_full'])
train_df, val_df = train_test_split_data(data)

N_EPOCHS = 10

BATCH_SIZE = 12


data_module = TripAdvisorDataModule(
    train_df,
    val_df,
    tokenizer,
    batch_size=BATCH_SIZE,
    max_token_len=400
)

steps_per_epoch = len(train_df) // BATCH_SIZE
total_training_steps = steps_per_epoch * N_EPOCHS
warmup_steps = total_training_steps // 5

print(f'labels: {sorted(train_df.rating_review.unique().tolist())}')

model = TripAdvisorClassifier(
  n_classes=train_df.rating_review.nunique(),
  n_warmup_steps=warmup_steps,
  n_training_steps=total_training_steps,
  model=BERT_MODEL_NAME,
)

checkpoint_callback = ModelCheckpoint(
  dirpath="checkpoints",
  filename="best-checkpoint",
  save_top_k=1,
  verbose=True,
  monitor="test_loss",
  mode="min"
)

logger = TensorBoardLogger("lightning_logs", name="trapadvisor")
early_stopping_callback = EarlyStopping(monitor='test_loss', patience=2)
trainer = pl.Trainer(
  logger=logger,
  callbacks=[early_stopping_callback, checkpoint_callback],
  max_epochs=N_EPOCHS,
  devices=1,
)
trainer.fit(model, data_module)
