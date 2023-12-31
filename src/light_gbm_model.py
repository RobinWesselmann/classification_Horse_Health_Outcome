from types import SimpleNamespace
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# credits: https://www.kaggle.com/code/ankurlimbashia/pg-s3e22-simple-lightgbm/notebook#Configuration

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, KFold
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import lightgbm as lgb
from tqdm import tqdm

import torch


load_dotenv(dotenv_path="../.env")
RAW_DATA_PATH = os.environ.get("RAW_DATA_PATH_py", "")
TRACKING_PATH = os.environ.get("TRACKING_PATH", "")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "")


def DataLoad(base_dir, f_name):
    data = pd.read_csv(base_dir + "/" + f_name)

    print(f"Shape of the {f_name.split('.')[0]} data: {data.shape}")
    return data

#############################
# Data Preprocessing


def split_lesion(x):
    l = len(x)
    if l == 4:
        return f'{x[0]},{x[1]},{x[2]},{x[3]}'
    elif l == 6:
        return f'{x[:2]},{x[2]},{x[3]},{x[-2:]}'
    elif l == 5 and x[:2] == '11' and x[-2:] == '10':
        if x[:2] == '11':
            return f'{x[:2]},{x[2]},{x[3]},{x[-1:]}'
        elif x[-2:] == '10':
            return f'{x[:1]},{x[2]},{x[3]},{x[-2:]}'
    elif l == 3:
        return f'0,{x[0]},{x[1]},{x[2]}'
    else:
        return '0,0,0,0'


class CustomLabelEncoder(LabelEncoder):
    def fit(self, y):
        super().fit(y)
        self.classes_ = np.append(self.classes_, '<unknown>')
        self.transform_mapping_ = {
            label: i for i, label in enumerate(self.classes_)}
        self.inverse_transform_mapping_ = {
            v: k for k, v in self.transform_mapping_.items()}

    def transform(self, y):
        return np.array([self.transform_mapping_.get(label, len(self.classes_) - 1) for label in y])

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        return np.array([self.inverse_transform_mapping_.get(label) for label in y])


class DataPrep:
    def __init__(self, df_train, df_test, cfg):
        self.df_train = df_train
        self.df_test = df_test
        self.cfg = cfg

    def preprocess(self,):
        sample_train = self.df_train[self.cfg.data["num_cols"] +
                                     self.cfg.data["cat_cols"] + self.cfg.data["target"]]
        sample_test = self.df_test[self.cfg.data["num_cols"] +
                                   self.cfg.data["cat_cols"]]

        sample_train = pd.concat([sample_train, sample_train.lesion_1.astype(
            "str").apply(split_lesion).str.split(',', expand=True).astype("int")], axis=1)
        sample_test = pd.concat([sample_test, sample_test.lesion_1.astype('str').apply(
            split_lesion).str.split(',', expand=True).astype('int')], axis=1)

        num_imputer = KNNImputer(n_neighbors=10,)
        cat_imputer = SimpleImputer(strategy="most_frequent")

        sample_train[self.cfg.data["cat_cols"]] = cat_imputer.fit_transform(
            sample_train[self.cfg.data["cat_cols"]])

        sample_train[self.cfg.data["num_cols"]] = num_imputer.fit_transform(
            sample_train[self.cfg.data["num_cols"]])

        sample_test[self.cfg.data['cat_cols']] = cat_imputer.fit_transform(
            sample_test[self.cfg.data['cat_cols']])
        sample_test[self.cfg.data['num_cols']] = num_imputer.fit_transform(
            sample_test[self.cfg.data['num_cols']])

        print("imputation completed", "."*10)

        for col in self.cfg.data["cat_cols"]:
            enc = CustomLabelEncoder()
            sample_train[col] = enc.fit_transform(sample_train[col])
            sample_test[col] = enc.transform(sample_test[col])

        print("encoding completed", "."*10)

        scaler = RobustScaler()
        sample_train[self.cfg.data["num_cols"]] = scaler.fit_transform(
            sample_train[self.cfg.data["num_cols"]])
        sample_test[self.cfg.data["num_cols"]] = scaler.transform(
            sample_test[self.cfg.data["num_cols"]])

        print("Scaling completed", "."*10)

        sample_train = sample_train.drop(self.cfg.data["rem_cols"], axis=1)
        sample_test = sample_test.drop(self.cfg.data["rem_cols"], axis=1)

        print("preprocessing completed", "."*10)
        print(
            f"Train Shape: {sample_train.shape} | Test Shape: {sample_test.shape}")
        return sample_train, sample_test

# Model Creation


def score_fn(y_pred, y_true):
    return f1_score(y_true, y_pred, average="micro")


def lgb_eval_fn(pred, eval_data):
    act = eval_data.get_label()
    pred = pred.reshape(act.shape[0], -1)
    return 'micro_f1_score', score_fn(pred.argmax(axis=1), act), True


class LGBModel:
    def __init__(self, cfg):
        self.weight_map = cfg.weight_map
        self.params = cfg.params_lgb
        self.target = "outcome"
        self.cat_cols = [c for c in cfg.data["cat_cols"]
                         if c not in cfg.data["rem_cols"]]
        self.features = None
        self.model = None

    def fit(self, df_train, df_val=None):
        self.features = df_train.drop(self.target, axis=1).columns.tolist()
        lgb_train = lgb.Dataset(
            data=df_train[self.features], label=df_train[self.target], weight=df_train[self.target].map(self.weight_map))

        if df_val is not None:
            lgb_val = lgb.Dataset(
                data=df_val[self.features], label=df_val[self.target], reference=lgb_train)
        params0 = {k: v for k, v in self.params.items()}
        self.model = lgb.train(
            params=params0, train_set=lgb_train,
            valid_sets=[lgb_val] if df_val is not None else None,
            # early_stopping_rounds=100 if df_val is not None else None,
            feval=lgb_eval_fn if df_val is not None else None,
            # verbose_eval=0 if df_val is not None else None
        )
        return self

    def predict(self, df_valid):
        return self.model.predict(df_valid[self.features])


if __name__ == "__main__":

    # load data
    submission_data = DataLoad(RAW_DATA_PATH, "sample_submission.csv")
    test_data = DataLoad(RAW_DATA_PATH, "test.csv")
    train_data = DataLoad(RAW_DATA_PATH, "train.csv")
    supplemental_data = DataLoad(RAW_DATA_PATH, "horse.csv")

    # join supplemental_data with train_data
    supplemental_data["id"] = train_data.shape[0] + \
        test_data.shape[0] + supplemental_data.index

    train_data = pd.concat([train_data, supplemental_data], axis=0)
    train_data = train_data.reset_index().drop(columns=["index"])

    # define configuration
    cfg = SimpleNamespace(**{})

    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu',
    cfg.device = cfg.device[0]
    cfg.SEED = 42

    cfg.data = {
        'num_cols': test_data.select_dtypes(include=['number']).columns.tolist(),
        'cat_cols': test_data.select_dtypes(exclude=['number']).columns.tolist(),
        'features': test_data.columns.drop(['id',]).tolist(),
        'target': ['outcome'],
        'rem_cols': ['id', 'age', 'lesion_3', 'lesion_2', 'lesion_1', 'hospital_number']
    }

    # lgbm params
    cfg.params_lgb = {
        'objective': 'multiclass',
        'num_class': 3,
        'boosting_type': 'gbdt',
        'num_leaves': 24,
        'max_depth': 10,
        'n_estimators': 450,
        'learning_rate': 0.08,
        'random_state': 42,
        'verbose': -1,
        'subsample': 0.8,
        'colsample_bytree': 0.65,
        'reg_alpha': 0.0001,
        'reg_lambda': 3.5,
        #     'metric':'cross_entropy',
    }

    cfg.weight_map = {0: 1.7, 1: 1.0, 2: 2.5}

    dataprep = DataPrep(train_data, test_data, cfg)
    df_train, df_test = dataprep.preprocess()

    enc = CustomLabelEncoder()
    df_train[cfg.data["target"][0]] = enc.fit_transform(
        df_train[cfg.data["target"][0]])

    kf = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=1, random_state=cfg.SEED)
    X = df_train.copy()
    y = df_train["outcome"]
    preds_val, acts_val, preds_test = [], [], []
    score = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[test_idx], y.iloc[test_idx]

        lgb_model = LGBModel(cfg)
        lgb_model = lgb_model.fit(X_train,)
        pred_val = lgb_model.predict(X_val).argmax(axis=1).tolist()

        scr = score_fn(pred_val, y_val)
        preds_val += pred_val
        acts_val += y_val.tolist()
        score.append(scr)

        pred_test = lgb_model.predict(df_test)
        preds_test.append(pred_test)

        print(f"Fold: {fold} | Score: {scr}")

    print(
        f"Score: {score_fn(preds_val, acts_val)} | mean_score: {np.mean(score)} | std of score: {np.std(score)}")

    lgb_model = LGBModel(cfg)
    lgb_model = lgb_model.fit(df_train)
    pred_test = lgb_model.predict(df_test).argmax(axis=1).tolist()

    # pred_test = np.mean(np.array(preds_test),axis = 0).argmax(axis = 1)
    pred_test = enc.inverse_transform(pred_test)

    submission_data['outcome'] = pred_test
    submission_data.to_csv(OUTPUT_PATH + "/submission_lgbm.csv", index=False)
