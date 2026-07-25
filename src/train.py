import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

from preprocessing import clean_text, LABEL_MAP

RANDOM_STATE = 42
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    train_df["Category"] = train_df["Class Index"].map(LABEL_MAP)
    test_df["Category"] = test_df["Class Index"].map(LABEL_MAP)
    return train_df, test_df


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    train_df, test_df = load_data()

    X_train = train_df["Description"].apply(clean_text)
    y_train = train_df["Category"]
    X_test = test_df["Description"].apply(clean_text)
    y_test = test_df["Category"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    report = classification_report(y_test, preds)

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(report)

    with open(REPORT_DIR / "metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nMacro F1: {macro_f1:.4f}\n\n{report}")

    cm = confusion_matrix(y_test, preds, labels=pipeline.classes_)
    ConfusionMatrixDisplay(cm, display_labels=pipeline.classes_).plot(xticks_rotation=45)
    plt.title("Confusion matrix — Logistic Regression")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png")

    joblib.dump(pipeline, MODEL_DIR / "model.pkl")
    print(f"\nSaved pipeline to {MODEL_DIR / 'model.pkl'}")


if __name__ == "__main__":
    main()