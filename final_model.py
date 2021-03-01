import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cat

from sklearn.model_selection import train_test_split, TimeSeriesSplit

print("Loading Data...")
train_data = pd.read_parquet("https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.parquet")
tournament_data = pd.read_parquet("https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.parquet")

features = [f for f in train_data.columns if f.startswith("feature")]
print(f"Loaded {len(features)} features")

folds = TimeSeriesSplit(n_splits=5)

print("Training LightGBM Model")

lgbm_parameters = {
    "num_leaves": 247,
    "min_child_weight": 0.04418018245999978,
    "feature_fraction": 0.6236531816961323,
    "bagging_fraction": 0.45905745759689537,
    "min_data_in_leaf": 2329,
    "objective": "regression",
    "max_depth": -1,
    "learning_rate": 0.005044370925360093,
    "boosting_type": "gbdt",
    "bagging_seed": 6,
    "verbosity": -1,
    "reg_alpha": 0.02650402149490516,
    "reg_lambda": 0.2710359329222913,
    "random_state": 0,
    "n_jobs": 1, # PLEASE DON'T CRASH
}

for fold_n, (train_index, valid_index) in enumerate(folds.split(train_data[features])):
    train_fold = lgb.Dataset(train_data[features].iloc[train_index],
                             train_data["target"].iloc[train_index])
    valid_fold = lgb.Dataset(train_data[features].iloc[valid_index],
                             train_data["target"].iloc[valid_index])
    reg = lgb.train(lgbm_parameters, train_fold, 10000,
                        valid_sets=[train_fold, valid_fold], verbose_eval=1000,
                        early_stopping_rounds=500)

del train_data
del train_fold
del valid_fold
del folds
del fold_n
del train_index
del valid_index

print("Generating predictions")
output = pd.DataFrame()
try:
#     output = tournament_data["id"]
#     output["prediction"] = reg.predict(tournament_data[features]) # PLEASE DON'T CRASH
#     output.to_csv("submission.csv", header=True, index=False)
    batch_size = 32_768
    for _, df in tournament_data.groupby(np.arange(len(tournament_data))//batch_size):
        print(df.head())
        result = pd.DataFrame()
        result["id"] = df["id"]
        result["prediction"] = reg.predict(df[features])
        output = output.append(result)
        del result
except:
    print("No more memory 😭")
output.to_csv("submission.csv", header=True, index=False)