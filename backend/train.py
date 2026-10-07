import sys
import os
import urllib.request
from pathlib import Path
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from backend.preprocess import clean
    from backend.features import get_vectorizer
except ImportError:
    from preprocess import clean
    from features import get_vectorizer

TRAIN_URL = "https://raw.githubusercontent.com/tfs4/liar_dataset/master/train.tsv"
TEST_URL = "https://raw.githubusercontent.com/tfs4/liar_dataset/master/test.tsv"

FAKE_LABELS = {"pants-fire", "false", "barely-true"}
REAL_LABELS = {"half-true", "mostly-true", "true"}

def download_file(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"Dataset download failed ({e}). Please specify a local dataset path.")
            sys.exit(1)
    return dest

def load_data(file_path: Path):
    df = pd.read_csv(file_path, sep="\t", header=None, usecols=[1, 2], dtype=str)
    df.columns = ["label", "statement"]
    df = df.dropna(subset=["label", "statement"])
    
    valid_mask = df["label"].isin(FAKE_LABELS | REAL_LABELS)
    df = df[valid_mask]
    
    # 0 for REAL, 1 for FAKE
    y = df["label"].apply(lambda x: 1 if x in FAKE_LABELS else 0).values
    texts = df["statement"].apply(clean).tolist()
    return texts, y

def main():
    data_dir = BACKEND_DIR / "data"
    train_path = download_file(TRAIN_URL, data_dir / "train.tsv")
    test_path = download_file(TEST_URL, data_dir / "test.tsv")
    
    X_train_raw, y_train = load_data(train_path)
    X_test_raw, y_test = load_data(test_path)
    
    vectorizer = get_vectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    joblib.dump({"vectorizer": vectorizer, "model": model}, BACKEND_DIR / "model.joblib")

if __name__ == "__main__":
    main()
