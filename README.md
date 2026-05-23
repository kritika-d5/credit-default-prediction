# Credit Default Prediction
 
ML model to predict credit defaults using the Give Me Some Credit dataset.
 
## Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
 
## Data
Download from: https://www.kaggle.com/c/GiveMeSomeCredit
Place `cs-training.csv` in `data/raw/`
 
## Project Structure
```
src/
  preprocess.py   # cleaning, SMOTE
  features.py     # feature engineering
  train.py        # model training
  evaluate.py     # metrics + plots
  explain.py      # SHAP
main.py           # full pipeline
```