# How to run
### Note: 
- All commands should be run from the root of the project
- Supported only Unix-like systems

## Install requirements
```
pip install -r requirements_nvidia.txt --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Load data from kaggle
```
cd trainig_process
./download_data.sh
```

## Train model
```
cd trainig_process
python3 main.py --help # to see all options
python3 main.py # to run with default options
``` 

## Run the FastAPI server
```