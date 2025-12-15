import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from pathlib import Path

DATA_PATH = Path("data/processed_data.csv")
MODEL_PATH = Path("model/churn_model.pkl")


def train_model():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_auc = roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1])

    # Train Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

    print(f"Logistic Regression ROC-AUC: {lr_auc:.3f}")
    print(f"Random Forest ROC-AUC: {rf_auc:.3f}")

    # Select best model
    best_model = rf_model if rf_auc > lr_auc else lr_model

    # Save best model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    print("Best model trained and saved successfully")


if __name__ == "__main__":
    train_model()
