import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import Trainer


class Pytorch_Lightning_MNIST_Classifier(pl.LightningModule):

    def __init__(self):
        super().__init__()
        # Задается архитектура нейросети
        self.layers = nn.Sequential(
            nn.Linear(28 * 28 * 1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        # Объявляется функция потерь
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layers(x)

    # Настраиваются параметры обучения
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        pred = self.layers(x)
        loss = self.loss_func(pred, y)
        self.log('train_loss', loss)
        return loss

    # Настраиваются параметры тестирования
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        pred = self.layers(x)
        loss = self.loss_func(pred, y)
        pred = torch.argmax(pred, dim=1)
        accuracy = torch.sum(y == pred).item() / (len(y) * 1.0)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', torch.tensor(accuracy), prog_bar=True)
        output = dict({
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy),
        })
        return output

    # Конфигурируется оптимизатор
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


transforms_set = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,),
                                                          (0.3081,))])

# Загрузка тренировочного датасета.
train_data = MNIST(root="data",
                   train=True,
                   download=True,
                   transform=transforms_set)

# Загрузка тестового датасета
test_data = MNIST(root="data",
                  train=False,
                  download=True,
                  transform=transforms_set)

# Инициализация модели и функции Trainer
Pytorch_lightning_MNIST_model = Pytorch_Lightning_MNIST_Classifier()
trainer = pl.Trainer(accelerator='gpu',
                     devices=1,
                     max_epochs=10)

# Обучение модели
trainer.fit(Pytorch_lightning_MNIST_model, DataLoader(train_data, batch_size=64, num_workers=32))

# Тестирование модели
trainer.test(Pytorch_lightning_MNIST_model, DataLoader(test_data, batch_size=64, num_workers=32))
