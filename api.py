import json
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import Depends, FastAPI
from pydantic import BaseModel

from model_training_process.main import PRE_TRAINED_MODEL_NAME, MAX_LEN, TOKENIZER
from model_training_process.tripadviser_classifier import TripAdvisorClassifier


class Predictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = TOKENIZER

        classifier = TripAdvisorClassifier(6, PRE_TRAINED_MODEL_NAME)
        classifier.load_state_dict(
            torch.load(Path('.') / 'model_training_process' / 'best_model_state.bin', map_location=self.device)
        )
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)

    def predict(self, text):
        encoded_text = self.tokenizer.encode_plus(
            text.lower(),
            max_length=MAX_LEN,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoded_text["input_ids"].to(self.device)
        attention_mask = encoded_text["attention_mask"].to(self.device)

        with torch.no_grad():
            probabilities = F.softmax(self.classifier(input_ids, attention_mask), dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        return {
            'predicted_class': predicted_class.cpu().item(),
            'confidence': confidence.cpu().item() * 100,
        }


app = FastAPI()


class Request(BaseModel):
    text: str


class Response(BaseModel):
    response: dict


def get_model():
    return Predictor()


@app.post("/predict", response_model=Response)
async def predict(request: Request, model: Predictor = Depends(get_model)):
    response = model.predict(request.text)
    print(response)
    return Response(
        response=response
    )
