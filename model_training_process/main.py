from collections import defaultdict

import click
import torch
from torch import nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from model_training_process.tripadviser_classifier import TripAdvisorClassifier
from model_training_process.utils import create_dataloaders, train_epoch, eval_model

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
RANDOM_SEED = 42
MAX_LEN = 128
BATCH_SIZE = 54
EPOCHS = 15
TOKENIZER = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


@click.command()
@click.option('--max_len', '-ml', default=MAX_LEN, help='maximum length of review text')
@click.option('--batch_size', '-bs', default=BATCH_SIZE, help='Batch size')
@click.option('--epochs', '-ep', default=EPOCHS, help='Number of epochs')
@click.option('--model_name', '-mn', default=PRE_TRAINED_MODEL_NAME,
              help='Name of model to use from transformers library')
@click.option('--random_seed', '-rs', default=RANDOM_SEED, help='Random seed number')
def run(max_len: int = MAX_LEN,
        batch_size: int = BATCH_SIZE,
        epochs: int = EPOCHS,
        model_name: str = PRE_TRAINED_MODEL_NAME,
        random_seed: int = RANDOM_SEED):
    """Main function to run model training process

    :param max_len: Maximum length of review text
    :param batch_size: Batch size
    :param epochs: Number of epochs
    :param model_name: Name of model to use from transformers library
    :param random_seed: Random seed number
    """
    train_data_loader, test_data_loader, val_data_loader, len_df_train, len_df_val = create_dataloaders(TOKENIZER,
                                                                                                        max_len,
                                                                                                        batch_size,
                                                                                                        random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TripAdvisorClassifier(n_classes=6, model_name=model_name)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_data_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(device)
    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(epochs):
        click.echo(f'Epoch {epoch + 1}/{epochs}')
        click.echo('-' * 10)
        train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler,
                                            len_df_train)
        click.echo(f'Train loss {train_loss} accuracy {train_acc}')
        val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len_df_val)
        click.echo(f'Val   loss {val_loss} accuracy {val_acc}')
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        if train_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = train_acc
