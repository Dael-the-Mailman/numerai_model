import os
import csv
import optuna
import pandas as pd
import numpy as np

from xgboost import XGBRegressor 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer

def correlation(targets, predictions):
    # ranked_preds = predictions.rank(pct=True, method="first")
    return np.corrcoef(predictions, targets)[0, 1]

def read_csv(file_path, reduce_mem=True):
    with open(file_path, 'r') as f:
        column_names = next(csv.reader(f))
        dtypes = {f"target": np.float16}
        to_uint8 = lambda x: np.uint8(float(x) * 4)
        converters = {x: to_uint8 for x in column_names if x.startswith('feature')}
        df = pd.read_csv(file_path, dtype=dtypes, converters=converters)

    return df

# def score(df, pred_name="target", target_name="prediction"):
#     return correlation(df[pred_name], df[target_name])

DIR = "E:/datasets/numerai_data/numerai_dataset_251"

print("Loading Data...")
train_data = read_csv(os.path.join(DIR, "numerai_training_data.csv"))
# tournament_data = read_csv(os.path.join(DIR, "numerai_tournament_data.csv"))

features = [f for f in train_data.columns if f.startswith("feature")]

print(f"Loaded {len(features)} features")

train_x, test_x, train_y, test_y = train_test_split(train_data[features], train_data["target"], test_size=0.15)
# val_x, test_x, val_y, test_y = train_test_split(split_x, split_y, test_size=0.4)

def objective(trial):
    parameters = {
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'learning_rate': trial.suggest_loguniform("learning_rate", 0.001, 0.5),
        'n_estimators': trial.suggest_int('n_estimators', 1000, 5000),
        'gpu_id': 0,
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'reg_alpha': trial.suggest_loguniform("reg_alpha", 1, 10),
        'reg_lambda': trial.suggest_loguniform("reg_lambda", 1, 10),
        'gamma': trial.suggest_loguniform("gamma", 1, 10),
        'colsample_bytree': trial.suggest_loguniform('colsample_bytree',0.1,1),
        'subsample': trial.suggest_loguniform('subsample',0.1,1),
        'min_child_weight': trial.suggest_loguniform('min_child_weight',1,10),
    }
    
    # eval_set = [(val_x, val_y)]

    model = XGBRegressor(**parameters)

    model.fit(train_x, train_y)

    predictions = model.predict(test_x)

    return correlation(test_y, predictions)

print("Finding best model...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))

hist = study.trials_dataframe()
print(hist.head())
hist.to_csv('xgbr_hist.csv')

# print("Finding best model...")
# reg = GridSearchCV(model, parameters, n_jobs=1, 
#                    scoring=make_scorer(correlation),
#                    verbose=2, refit=True)
# reg.fit(train_data[features], train_data["target"])
# print('Best Model:', reg.best_estimator_)
# reg.save_model("XGBRegressor.xgb")

# print("Generating predictions")
# tournament_data["prediction"] = reg.predict(tournament_data[features])
# tournament_data["prediction"].to_csv("xgbr_submission.csv", header=True)

print("Success 😎")