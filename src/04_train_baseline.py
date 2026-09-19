import os

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


DATA_DIR = "data/processed"
MODEL_DIR = "models"

RANDOM_STATE = 42


def load_data():

    print("\n========== LOADING TRAINING DATA ==========\n")

    X_train = np.load(
        f"{DATA_DIR}/X_train.npy"
    )

    X_val = np.load(
        f"{DATA_DIR}/X_val.npy"
    )

    y_train = np.load(
        f"{DATA_DIR}/y_train.npy"
    )

    y_val = np.load(
        f"{DATA_DIR}/y_val.npy"
    )

    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)

    return X_train, X_val, y_train, y_val


def train_model(X_train, y_train):

    print("\n========== TRAINING RANDOM FOREST ==========\n")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    print("Training complete.")

    return model


def evaluate_model(model, X_val, y_val):

    print("\n========== EVALUATION ==========\n")

    probabilities = model.predict_proba(X_val)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(np.uint8)

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_val,
            predictions,
            digits=4
        )
    )

    print("Confusion Matrix:\n")

    print(
        confusion_matrix(
            y_val,
            predictions
        )
    )

    roc_auc = roc_auc_score(
        y_val,
        probabilities
    )

    pr_auc = average_precision_score(
        y_val,
        probabilities
    )

    print("\nROC-AUC:", roc_auc)

    print("PR-AUC:", pr_auc)

    return probabilities


def feature_importance(model):

    print("\n========== FEATURE IMPORTANCE ==========\n")

    feature_names = [
        "mag_anom",
        "rtp",
        "tmi_hg",
        "geod_2ndinv",
        "iso_grav_anom_slope",
        "tc",
        "geod_shearrate",
        "geod_dilaterate",
        "tmi_vg",
        "deq_n100a15",
        "iso_grav_anom_vg",
        "det_elev",
        "iso_grav_anom",
        "tmi",
        "depth_to_base_surf",
        "ieq_n100a15",
        "cond_surf",
        "iso_grav_anom_hg",
        "det_elev_slope"
    ]

    importances = model.feature_importances_

    results = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    for name, importance in results:

        print(
            f"{name:25s} "
            f"{importance:.6f}"
        )


def save_model(model):

    import joblib

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    path = (
        f"{MODEL_DIR}/"
        "random_forest_baseline.joblib"
    )

    joblib.dump(
        model,
        path
    )

    print("\nModel saved to:")
    print(path)


def main():

    X_train, X_val, y_train, y_val = load_data()

    model = train_model(
        X_train,
        y_train
    )

    evaluate_model(
        model,
        X_val,
        y_val
    )

    feature_importance(
        model
    )

    save_model(
        model
    )


if __name__ == "__main__":
    main()