import os, hmac, httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="TITAN Ω Cloud API", version="0.1.0")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
TITAN_API_TOKEN = os.getenv("TITAN_API_TOKEN", "")

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=30)

class ChatResponse(BaseModel):
    reply: str
    model: str

@app.get("/health")
async def health():
    return {"status": "ok", "service": "TITAN Ω Cloud API"}

def authorize(header):
    if not TITAN_API_TOKEN:
        raise HTTPException(503, "Cloud gateway is not configured")
    if not header or not hmac.compare_digest(header, f"Bearer {TITAN_API_TOKEN}"):
        raise HTTPException(401, "Unauthorized")

@app.post("/v1/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, authorization: str | None = Header(default=None)):
    authorize(authorization)
    if not GEMINI_API_KEY:
        raise HTTPException(503, "AI provider key is not configured")
    contents = []
    for item in body.history[-20:]:
        role, value = item.get("role"), item.get("text", "")
        if role in ("user", "model") and isinstance(value, str) and value.strip():
            contents.append({"role": role, "parts": [{"text": value[:12000]}]})
    contents.append({"role": "user", "parts": [{"text": body.message}]})
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(url, params={"key": GEMINI_API_KEY},
                json={"contents": contents, "generationConfig": {"temperature": 0.7, "maxOutputTokens": 4096}})
        if r.status_code >= 400:
            raise HTTPException(502, "AI provider request failed")
        data = r.json()
        candidates = data.get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        reply = "\\n".join(p.get("text", "") for p in parts if isinstance(p, dict))
        if not reply.strip():
            raise HTTPException(502, "AI provider returned an empty response")
        return ChatResponse(reply=reply, model=GEMINI_MODEL)
    except httpx.TimeoutException:
        raise HTTPException(504, "AI provider timed out")
    except httpx.HTTPError:
        raise HTTPException(502, "Could not reach AI provider")
