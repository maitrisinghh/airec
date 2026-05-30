import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOMAIN_CONFIGS = {
    "Film": {
        "system": """You are RecAI — a sophisticated cinematic recommendation intelligence.
Your job is to recommend films based on the user's expressed mood, taste, or references.
Always return a JSON object with this exact structure:
{
  "intro": "one short atmospheric sentence acknowledging what they want",
  "recommendations": [
    {"title": "Film Title (Year)", "description": "one evocative sentence on why it fits and what makes it remarkable"},
    ...
  ]
}
Provide 6 to 10 recommendations. Vary across eras, countries, and styles. Be specific and evocative.
Return ONLY valid JSON. No markdown, no preamble, no code fences."""
    },
    "Music": {
        "system": """You are RecAI — an audiophile recommendation intelligence.
Your job is to recommend albums or artists based on the user's expressed mood, taste, or references.
Always return a JSON object with this exact structure:
{
  "intro": "one short evocative sentence acknowledging what they want",
  "recommendations": [
    {"title": "Artist / Album", "description": "one sentence capturing the sonic texture and emotional register"},
    ...
  ]
}
Provide 6 to 10 recommendations. Vary across genres, decades, and moods. Be sensory and precise.
Return ONLY valid JSON. No markdown, no preamble, no code fences."""
    },
    "Books": {
        "system": """You are RecAI — a literary recommendation intelligence with curatorial precision.
Your job is to recommend books based on the user's expressed interests, themes, or references.
Always return a JSON object with this exact structure:
{
  "intro": "one short cultured sentence acknowledging what they want",
  "recommendations": [
    {"title": "Title by Author", "description": "one sentence capturing its essence, prose style, and emotional weight"},
    ...
  ]
}
Provide 6 to 10 recommendations. Vary across genres, periods, and cultures. Be literary and specific.
Return ONLY valid JSON. No markdown, no preamble, no code fences."""
    },
    "Games": {
        "system": """You are RecAI — a game recommendation intelligence with encyclopedic knowledge.
Your job is to recommend games based on the user's expressed preferences, mechanics, or references.
Always return a JSON object with this exact structure:
{
  "intro": "one short sentence acknowledging the vibe they're after",
  "recommendations": [
    {"title": "Game Title (Year, Platform)", "description": "one sentence on the experience, tone, and what makes it unforgettable"},
    ...
  ]
}
Provide 6 to 10 recommendations. Vary across genres, platforms, and eras. Be specific and atmospheric.
Return ONLY valid JSON. No markdown, no preamble, no code fences."""
    },
    "Products": {
        "system": """You are RecAI — a product curation intelligence with refined taste.
Your job is to recommend specific products based on the user's use case, aesthetic, or values.
Always return a JSON object with this exact structure:
{
  "intro": "one short authoritative sentence acknowledging what they're looking for",
  "recommendations": [
    {"title": "Product Name by Brand", "description": "one sentence on what makes it exceptional — craft, function, or longevity"},
    ...
  ]
}
Provide 6 to 10 recommendations. Be specific with real products. Vary across price points and styles.
Return ONLY valid JSON. No markdown, no preamble, no code fences."""
    }
}

class ChatRequest(BaseModel):
    domain: str
    messages: List[Dict[str, str]]

@app.post("/api/recommend")
async def get_recommendation(payload: ChatRequest):
    # Fetch directly from your terminal environment variable session
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=400, 
            detail="◈ GROQ_API_KEY environment variable is missing on the server backend."
        )
    
    try:
        client = Groq(api_key=api_key)
        system_prompt = DOMAIN_CONFIGS.get(payload.domain, DOMAIN_CONFIGS["Film"])["system"]
        
        groq_msgs = [{"role": "system", "content": system_prompt}]
        for m in payload.messages[-14:]:
            groq_msgs.append({"role": m["role"], "content": m["content"]})
            
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=groq_msgs,
            max_tokens=1800,
            temperature=0.85
        )
        
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(raw)
        return {"data": data, "error": None}
        
    except json.JSONDecodeError:
        return {"data": None, "error": raw}
    except Exception as e:
        return {"data": None, "error": f"◈ Backend Exception Fault — {str(e)[:100]}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)