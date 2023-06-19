# MIPT final project:
#### Theme rus: Разработка и применение нейронных сетей трансформеров на основе BERT архитектуры для классификации текста с использованием PyTorch, FastAPI и Docker.
#### Theme eng: Text Classification using Transformer Neural Networks, with BERT Architecture, using PyTorch, FastAPI, and Docker.
## How to run.
### Note: 
- All commands should be run from the root of the project
- Supported only Unix-like systems

## Install requirements.
```
pip install -r requirements_nvidia.txt --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Load data from kaggle.
```
./get_kaggle_data.sh
```

## Train model.
```
python3 train_model.py --help # To see all options
python3 train_model.py # To run with default options
``` 

## Run the FastAPI server in docker container.
```
docker build -t fastapi:api .
docker run -p 8001:8001 fastapi:api
```

## Test, how it works.
```
import requests
import json

r = requests.post('http://127.0.0.1:8001/predict', 
                  data=json.dumps({'text': 'I hate this hotel!'}))
print(r.json())
```

