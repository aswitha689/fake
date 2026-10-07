import sys
from pathlib import Path
from lime.lime_text import LimeTextExplainer

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from backend.classifier import get_artifacts
    from backend.preprocess import clean
except ImportError:
    from classifier import get_artifacts
    from preprocess import clean

_EXPLAINER = None

def get_explainer() -> LimeTextExplainer:
    global _EXPLAINER
    if _EXPLAINER is None:
        _EXPLAINER = LimeTextExplainer(class_names=["REAL", "FAKE"], random_state=42)
    return _EXPLAINER

def _fallback_explain(text: str, vectorizer, model, top_n: int = 8) -> list[dict]:
    cleaned = clean(text)
    words = list(dict.fromkeys(cleaned.split()))
    vocab = vectorizer.vocabulary_
    coefs = model.coef_[0]
    scored = []
    for w in words:
        if w in vocab:
            idx = vocab[w]
            val = float(coefs[idx])
            scored.append((w, val))
    scored.sort(key=lambda x: abs(x[1]), reverse=True)
    return [
        {
            "word": str(w),
            "weight": round(float(val), 4),
            "direction": "FAKE" if val > 0 else "REAL"
        }
        for w, val in scored[:top_n]
    ]

def explain(text: str, top_n: int = 8, num_samples: int = 500) -> list[dict]:
    artifacts = get_artifacts()
    vectorizer = artifacts["vectorizer"]
    model = artifacts["model"]

    def predict_proba(texts):
        cleaned_list = [clean(t) for t in texts]
        vecs = vectorizer.transform(cleaned_list)
        return model.predict_proba(vecs)

    try:
        explainer = get_explainer()
        exp = explainer.explain_instance(
            text,
            predict_proba,
            labels=[1],
            num_features=top_n,
            num_samples=num_samples
        )
        lime_list = exp.as_list(label=1)
        results = []
        for word, weight in lime_list:
            w_float = float(weight)
            results.append({
                "word": str(word),
                "weight": round(w_float, 4),
                "direction": "FAKE" if w_float > 0 else "REAL"
            })
        return results
    except Exception:
        return _fallback_explain(text, vectorizer, model, top_n=top_n)
