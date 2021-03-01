import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold

def correlation(targets, predictions):
    return np.corrcoef(predictions, targets)[0, 1]

print("Loading Data...")

train_data = pd.read_parquet("https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.parquet")

features = [f for f in train_data.columns if f.startswith("feature")]

print(f"Loaded {len(features)} features")

folds = TimeSeriesSplit(n_splits=5)
x_train, x_test, y_train, y_test = train_test_split(train_data[features], train_data["target"], test_size=0.15)

def objective(trial):
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = features
    errors = list()
    
    parameters = {
        "num_leaves": trial.suggest_int("num_leaves", 64, 256),
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 1e-3, 0.06),
        "feature_fraction": trial.suggest_loguniform("feature_fraction", 0.01, 1.0),
        "bagging_fraction": trial.suggest_loguniform("bagging_fraction", 0.01, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 5000),
        "objective": "regression",
        "max_depth": -1,
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.1),
        "boosting_type": "gbdt",
        "bagging_seed": trial.suggest_int("bagging_seed", 1, 10),
        "verbosity": -1,
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10),
        "random_state": 0
    }

    for fold_n, (train_index, valid_index) in enumerate(folds.split(x_train)):
        train_fold = lgb.Dataset(x_train.iloc[train_index], 
                                 y_train.iloc[train_index])
        valid_fold = lgb.Dataset(x_train.iloc[valid_index],
                                 y_train.iloc[valid_index])
        reg = lgb.train(parameters, train_fold, 10000,
                        valid_sets=[train_fold, valid_fold], verbose_eval=1000,
                        early_stopping_rounds=500)
        feature_importances['fold_{}'.format(fold_n+1)] = reg.feature_importance()
    
    predictions = reg.predict(x_test)
    
    return correlation(y_test, predictions)

print("Finding best model...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))

hist = study.trials_dataframe()
print(hist.head())
hist.to_csv('xgbr_hist.csv')