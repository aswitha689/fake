# Explainable Fake News Detection System

1. Install dependencies:
   pip install -r requirements.txt
2. Train ML pipeline:
   python -m backend.train
3. (Optional) Configure Google Fact Check API key:
   export FACTCHECK_API_KEY="your_key"  # Windows PowerShell: $env:FACTCHECK_API_KEY="your_key"
4. Start backend server:
   python -m uvicorn backend.app:app --port 8000
5. Open frontend:
   Open frontend/index.html in your browser.
