# Facial Expression Analyzer (FEA)

## Data
### Datasets
Extract the datasets after a commit:
```tar -xvzf train/datasets.gzip```

Compress the datasets before a commit:
```tar -cz train/Datasets/ > train/datasets.gzip```

### Test pics
Extract the datasets after a commit:
```tar -xvzf test/test_pics.gzip```

Compress the datasets before a commit:
```tar -cz test/Test_pics/ > test/test_pics.gzip```

## Google Colab
### First time
1. Go to https://colab.research.google.com/github/ and select `"Include private repos"`
2. Under Repository: pick `SHAHMIRALI/Facial-Expression-Analyzer`
3. Under Branch: pick `main`
1. Open `train.ipynb`.
2. Open `predict.ipynb`.

### Notebooks
`train.ipynb` Colab notebook: https://colab.research.google.com/github/420NPEasy/FEA/blob/main/train.ipynb

`predict.ipynb` Colab notebook: https://colab.research.google.com/github/420NPEasy/FEA/blob/main/predict.ipynb

## Code
Python version: 3

### Basic requirements
`python -m pip install -r requirements.txt`
Install packages required to use model for predictions

### Extras
`python -m pip install -r requirements_extra.txt`
Install extra packages required to train a model
