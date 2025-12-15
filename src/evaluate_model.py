import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_PATH = Path("data/processed_data.csv")
MODEL_PATH = Path("model/churn_model.pkl")


def evaluate_model():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.3f}")


if __name__ == "__main__":
    evaluate_model()

