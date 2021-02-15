import os
import pandas as pd

from xgboost import XGBRegressor
from utils import read_csv

DIR = "E:/datasets/numerai_data/numerai_dataset_251"

print("Loading Data...")
training_data = read_csv(os.path.join(DIR, "numerai_training_data.csv"), reduce_mem=True)
tournament_data = read_csv(os.path.join(DIR, "numerai_tournament_data.csv"), reduce_mem=True)
features = [f for f in tournament_data.columns if f.startswith("feature")]

print(f"Loaded {len(features)} features")

parameters = {
    'max_depth': 6,
    'learning_rate': 0.0455931325466394,
    'n_estimators': 1880,
    # 'gpu_id': 0,
    # 'tree_method': 'gpu_hist',
    # 'predictor': 'gpu_predictor',
    'reg_alpha': 9.21300235811859,
    'reg_lambda': 7.67094310505669,
    'gamma': 1.25304737179889,
    'colsample_bytree': 0.176873083338503,
    'subsample': 0.326513157234183,
    'min_child_weight': 1.06868133069853,
}

model = XGBRegressor(**parameters)

print("Training Model...")
model.fit(training_data[features], training_data["target"])
# pickle.dump(model,open('xgbr.model', 'wb'))

print("Generating predictions")
tournament_data["prediction"] = model.predict(tournament_data[features])
tournament_data[["id","prediction"]].to_csv("submission.csv", header=True, index=False)