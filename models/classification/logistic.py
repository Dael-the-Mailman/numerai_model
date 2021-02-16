import os
import csv
import optuna
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def correlation(targets, predictions):
    # ranked_preds = predictions.rank(pct=True, method="first")
    return np.corrcoef(predictions, targets)[0, 1]

def read_csv(file_path):
    with open(file_path, 'r') as f:
        column_names = next(csv.reader(f))
        # dtypes = {f"target": np.float16}
        to_uint8 = lambda x: np.uint8(float(x) * 4)
        converters = {x: to_uint8 for x in column_names if x.startswith('feature') or x.startswith('target')}
        df = pd.read_csv(file_path, converters=converters)

    return df

print("Loading Data...")
train_data = read_csv('E:\\datasets\\numerai_data\\numerai_dataset_251\\numerai_training_data.csv')
features = [f for f in train_data.columns if f.startswith("feature")]
print(f"Loaded {len(features)} features")

train_x, test_x, train_y, test_y = train_test_split(train_data[features], train_data["target"], test_size=0.15)

def objective(trial):
    parameters = {
        'penalty': 'elasticnet',
        'tol': trial.suggest_loguniform('tol', 1e-4, 0.1),
        'C': trial.suggest_loguniform('C', 1, 100),
        'solver': 'saga',
        'max_iter': trial.suggest_int('max_iter', 50, 1000),
        'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
        'n_jobs': 1
    }

    model = LogisticRegression(**parameters)

    model.fit(train_x, train_y)

    predictions = model.predict(test_x)
    predictions = predictions / 4

    return correlation(test_y, predictions)

print("Finding best model...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))

hist = study.trials_dataframe()
print(hist.head())
hist.to_csv('logistic_hist.csv')
# print(train_data.info())