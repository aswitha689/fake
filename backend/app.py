import sys
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from backend.db import get_db, init_db, get_iso_now
    from backend.classifier import predict as run_predict
    from backend.explain import explain as run_explain
    from backend.grounding import check_claim
except ImportError:
    from db import get_db, init_db, get_iso_now
    from classifier import predict as run_predict
    from explain import explain as run_explain
    from grounding import check_claim

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Fake News Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    prediction_id: int
    user_label: str
    comment: Optional[str] = ""

@app.get("/")
def index():
    return FileResponse(PROJECT_DIR / "frontend" / "index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 1. Submitted
        now = get_iso_now()
        cursor.execute(
            "INSERT INTO Statement (text, submitted_at, status) VALUES (?, ?, ?)",
            (text, now, "Submitted")
        )
        statement_id = cursor.lastrowid
        
        # 2. Classified
        label, confidence = run_predict(text)
        cursor.execute(
            "UPDATE Statement SET status = ? WHERE id = ?",
            ("Classified", statement_id)
        )
        cursor.execute(
            "INSERT INTO Prediction (statement_id, label, confidence) VALUES (?, ?, ?)",
            (statement_id, label, confidence)
        )
        prediction_id = cursor.lastrowid
        
        # 3. Explained
        explanation = run_explain(text)
        for item in explanation:
            cursor.execute(
                "INSERT INTO Explanation (prediction_id, word, weight, direction) VALUES (?, ?, ?, ?)",
                (prediction_id, item["word"], item["weight"], item["direction"])
            )
        
        # 4. Grounding
        grounding = check_claim(text)
        if grounding.get("matches"):
            for m in grounding["matches"]:
                cursor.execute(
                    "INSERT INTO KnowledgeFact (statement_id, claim, rating, publisher, url) VALUES (?, ?, ?, ?, ?)",
                    (statement_id, m.get("claim", ""), m.get("rating", ""), m.get("publisher", ""), m.get("url", ""))
                )
        
        # Status branch: confidence < 0.60 -> "Needs Review", else "Explained"
        final_status = "Needs Review" if confidence < 0.60 else "Explained"
        cursor.execute(
            "UPDATE Statement SET status = ? WHERE id = ?",
            (final_status, statement_id)
        )
        conn.commit()

    return {
        "statement_id": statement_id,
        "prediction_id": prediction_id,
        "label": label,
        "confidence": confidence,
        "status": final_status,
        "explanation": explanation,
        "grounding": grounding
    }

@app.get("/history")
def get_history():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                s.id AS statement_id,
                s.text,
                s.submitted_at,
                s.status,
                p.id AS prediction_id,
                p.label,
                p.confidence
            FROM Statement s
            LEFT JOIN Prediction p ON p.statement_id = s.id
            ORDER BY s.id DESC
            LIMIT 20
        """)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

@app.get("/statement/{id}")
def get_statement(id: int):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Statement WHERE id = ?", (id,))
        statement = cursor.fetchone()
        if not statement:
            raise HTTPException(status_code=404, detail="Statement not found.")
        
        cursor.execute("SELECT * FROM Prediction WHERE statement_id = ? ORDER BY id DESC LIMIT 1", (id,))
        prediction = cursor.fetchone()
        
        explanations = []
        feedbacks = []
        if prediction:
            cursor.execute(
                "SELECT word, weight, direction FROM Explanation WHERE prediction_id = ?",
                (prediction["id"],)
            )
            explanations = [dict(r) for r in cursor.fetchall()]
            
            cursor.execute(
                "SELECT id, user_label, comment, created_at FROM Feedback WHERE prediction_id = ?",
                (prediction["id"],)
            )
            feedbacks = [dict(r) for r in cursor.fetchall()]
            
        cursor.execute("SELECT claim, rating, publisher, url FROM KnowledgeFact WHERE statement_id = ?", (id,))
        facts = [dict(r) for r in cursor.fetchall()]
        
        return {
            "statement_id": statement["id"],
            "text": statement["text"],
            "submitted_at": statement["submitted_at"],
            "status": statement["status"],
            "prediction_id": prediction["id"] if prediction else None,
            "label": prediction["label"] if prediction else None,
            "confidence": prediction["confidence"] if prediction else None,
            "explanation": explanations,
            "grounding": {
                "available": len(facts) > 0,
                "matches": facts
            },
            "feedbacks": feedbacks
        }

@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM Prediction WHERE id = ?", (req.prediction_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Prediction not found.")
        
        now = get_iso_now()
        cursor.execute(
            "INSERT INTO Feedback (prediction_id, user_label, comment, created_at) VALUES (?, ?, ?, ?)",
            (req.prediction_id, req.user_label, req.comment or "", now)
        )
        conn.commit()
        return {"status": "success", "id": cursor.lastrowid}

@app.delete("/statement/{id}")
def delete_statement(id: int):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM Statement WHERE id = ?", (id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Statement not found.")
        
        cursor.execute("DELETE FROM Feedback WHERE prediction_id IN (SELECT id FROM Prediction WHERE statement_id = ?)", (id,))
        cursor.execute("DELETE FROM Explanation WHERE prediction_id IN (SELECT id FROM Prediction WHERE statement_id = ?)", (id,))
        cursor.execute("DELETE FROM Prediction WHERE statement_id = ?", (id,))
        cursor.execute("DELETE FROM KnowledgeFact WHERE statement_id = ?", (id,))
        cursor.execute("DELETE FROM Statement WHERE id = ?", (id,))
        conn.commit()
        return {"status": "deleted", "id": id}
