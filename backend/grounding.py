import os
import requests

FACT_CHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

def check_claim(query: str) -> dict:
    api_key = os.getenv("FACTCHECK_API_KEY")
    if not api_key or not query or not query.strip():
        return {"available": False, "matches": []}
    
    try:
        response = requests.get(
            FACT_CHECK_API_URL,
            params={
                "query": query.strip(),
                "key": api_key,
                "pageSize": 3
            },
            timeout=5
        )
        if response.status_code != 200:
            return {"available": False, "matches": []}
        
        data = response.json()
        claims = data.get("claims", [])
        if not claims:
            return {"available": False, "matches": []}
        
        matches = []
        for item in claims[:3]:
            claim_text = item.get("text", "")
            reviews = item.get("claimReview", [])
            rating = ""
            publisher = ""
            url = ""
            if reviews:
                first_review = reviews[0]
                rating = first_review.get("textualRating", "")
                publisher = first_review.get("publisher", {}).get("name", "")
                url = first_review.get("url", "")
            
            matches.append({
                "claim": claim_text,
                "rating": rating,
                "publisher": publisher,
                "url": url
            })
            
        return {
            "available": True,
            "matches": matches
        }
    except Exception:
        return {"available": False, "matches": []}
