from torch import nn
from transformers import BertModel


class TripAdvisorClassifier(nn.Module):
    """Class for TripAdvisor classifier model

    :param n_classes: number of classes to predict
    :param model_name: name of model to use from transformers library
    """
    def __init__(self, n_classes: int, model_name: str):
        super(TripAdvisorClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name, return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)