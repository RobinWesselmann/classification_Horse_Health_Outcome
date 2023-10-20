import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow as mlf
from mlflow.data.pandas_dataset import PandasDataset

from sklearn.model_selection import StratifiedKFold, cross_validate

load_dotenv(dotenv_path="../.env")
RAW_DATA_PATH = os.environ.get("RAW_DATA_PATH_py", "")
TRACKING_PATH = os.environ.get("TRACKING_PATH", "")
PULSE_ENCODING = {"yes": 1, "no": 0}
OUTCOME_ENCODING = {"died": 0, "euthanized": 1, "lived": 2}


def print_experiment_infos(experiment):
    print(f"Name: {experiment.name}")
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tags: {experiment.tags}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")


if __name__ == "__main__":

    # load data
    submission_data = pd.read_csv(RAW_DATA_PATH + "/sample_submission.csv")
    test_data = pd.read_csv(RAW_DATA_PATH + "/test.csv")
    train_data = pd.read_csv(RAW_DATA_PATH + "/train.csv")

    # setup tracking
    tracking_path = TRACKING_PATH + "/mlruns"
    mlflow_client = mlf.MlflowClient(tracking_uri=tracking_path)

    try:
        experiment_id = mlflow_client.create_experiment(
            name="test_experiment", artifact_location=tracking_path)
        experiment = mlflow_client.get_experiment(experiment_id)
        print_experiment_infos(experiment)

    except Exception as e:
        print(e)
        experiment = mlflow_client.get_experiment("961575633781897667")
        print_experiment_infos(experiment)

    # separate data
    test_feat = ["pulse", "respiratory_rate",
                 "packed_cell_volume", "surgery"]
    train_feat = test_feat + ["outcome"]

    train, test = train_data[train_feat], test_data[test_feat]

    # Train
    run_name = "test_run"
    run = mlflow_client.create_run(
        experiment_id=experiment.experiment_id, run_name=run_name)

    params = {
        "n_estimators": 500,
        "criterion": "gini",
        "max_depth": 8
    }

    kf = StratifiedKFold(n_splits=5, shuffle=False)

    metrics = {}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=train, y=train["outcome"])):
        print(f"{10*'#'} Fold {fold} {10*'#'}")

        train_tmp, valid_tmp = train.iloc[train_idx], train.iloc[val_idx]
        X_train, y_train = train_tmp.drop(
            "outcome", axis=1), train_tmp["outcome"]
        X_valid, y_valid = valid_tmp.drop(
            "outcome", axis=1), valid_tmp["outcome"]

        # prepare train
        X_train["surgery"] = X_train["surgery"].map(PULSE_ENCODING)
        y_train = y_train.map(OUTCOME_ENCODING)

        # train
        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)

        # prepare valid
        X_valid["surgery"] = X_valid["surgery"].map(PULSE_ENCODING)
        y_valid = y_valid.map(OUTCOME_ENCODING)
        y_pred = rf.predict(X_valid)

        acc_score = accuracy_score(y_true=y_valid, y_pred=y_pred)
        precision = precision_score(
            y_true=y_valid, y_pred=y_pred, average="macro")
        recall = recall_score(y_true=y_valid, y_pred=y_pred, average="macro")

        for score_name, score in zip(["accuracy", "precision", "recall"], [acc_score, precision, recall]):
            print(f"Fold({fold}) {score_name}: {score}\n")

            if metrics.get(score_name) is None:
                metrics[score_name] = [score]
            else:
                metrics[score_name].append(score)

    mlflow_client.log_metric(
        run_id=run.info.run_id, key="accuracy_cv", value=np.mean(metrics["accuracy"]))
    mlflow_client.log_metric(
        run_id=run.info.run_id, key="precision_cv", value=np.mean(metrics["precision"]))
    mlflow_client.log_metric(run_id=run.info.run_id,
                             key="recall_cv", value=np.mean(metrics["recall"]))

    mlflow_client.log_param(run_id=run.info.run_id,
                            key="Feature Names", value=train_feat)

    X_train.to_csv("X_train.csv")
    mlflow_client.log_artifact(
        run_id=run.info.run_id, local_path="X_train.csv")

    dataset = mlflow_client.data.from_pandas(train, source="train.csv")
    mlflow_client.log_inputs(run.info.run_id, dataset)

    model_params = rf.get_params()

    for param in list(model_params.keys()):
        mlflow_client.log_param(run_id=run.info.run_id,
                                key=f"{param}", value=model_params[param])

    mlflow_client.update_run(run.info.run_id, "FINISHED")
