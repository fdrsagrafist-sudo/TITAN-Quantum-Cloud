import os
import hmac
import httpx

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field


app = FastAPI(
    title="TITAN Ω Cloud API",
    version="0.3.0"
)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
TITAN_API_TOKEN = os.getenv("TITAN_API_TOKEN", "")


api_key_header = APIKeyHeader(
    name="Authorization",
    scheme_name="TITAN_Bearer_Auth",
    auto_error=False
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    history: list[dict[str, str]] = Field(
        default_factory=list,
        max_length=30
    )


class ChatResponse(BaseModel):
    reply: str
    model: str


def authorize(token: str | None):
    if not TITAN_API_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Cloud gateway is not configured"
        )

    expected = f"Bearer {TITAN_API_TOKEN}"

    if not token or not hmac.compare_digest(token, expected):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )


@app.get("/")
async def root():
    return {
        "service": "TITAN Ω Cloud API",
        "status": "online",
        "version": "0.3.0"
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "TITAN Ω Cloud API"
    }


@app.get("/v1/diagnostics")
async def diagnostics(
    authorization: str | None = Depends(api_key_header)
):
    authorize(authorization)

    if not GEMINI_API_KEY:
        return {
            "status": "error",
            "reason": "GEMINI_API_KEY is missing",
            "model": GEMINI_MODEL
        }

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{GEMINI_MODEL}:generateContent"
    )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                url,
                headers={
                    "x-goog-api-key": GEMINI_API_KEY
                },
                json={
                    "contents": [{
                        "role": "user",
                        "parts": [{"text": "Reply with OK"}]
                    }]
                }
            )

        if response.status_code != 200:
            try:
                error_data = response.json().get("error", {})
                message = error_data.get(
                    "message",
                    "Unknown provider error"
                )
                reason = (
                    error_data.get("details", [{}])[0]
                    .get("reason", "Not provided")
                )
            except Exception:
                message = "Unparseable provider response"
                reason = "Unknown"

            return {
                "status": "failed",
                "http_status": response.status_code,
                "reason": reason,
                "message": message,
                "model": GEMINI_MODEL
            }

        return {
            "status": "success",
            "gemini_connected": True,
            "model": GEMINI_MODEL
        }

    except httpx.TimeoutException:
        return {
            "status": "timeout",
            "model": GEMINI_MODEL
        }

    except httpx.HTTPError:
        return {
            "status": "connection_error",
            "model": GEMINI_MODEL
        }


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    authorization: str | None = Depends(api_key_header)
):
    authorize(authorization)

    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="AI provider key is not configured"
        )

    contents = []

    for item in body.history[-20:]:
        role = item.get("role")
        value = item.get("text", "")

        if (
            role in ("user", "model")
            and isinstance(value, str)
            and value.strip()
        ):
            contents.append({
                "role": role,
                "parts": [{"text": value[:12000]}]
            })

    contents.append({
        "role": "user",
        "parts": [{"text": body.message}]
    })

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{GEMINI_MODEL}:generateContent"
    )

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                url,
                headers={
                    "x-goog-api-key": GEMINI_API_KEY
                },
                json={
                    "contents": contents,
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 4096
                    }
                }
            )

        if response.status_code >= 400:
            print("GEMINI ERROR STATUS:", response.status_code)
            print("GEMINI ERROR BODY:", response.text[:2000])

            raise HTTPException(
                status_code=502,
                detail="AI provider request failed"
            )

        data = response.json()
        candidates = data.get("candidates") or []

        parts = (
            candidates[0].get("content", {}).get("parts", [])
            if candidates else []
        )

        reply = "\n".join(
            part.get("text", "")
            for part in parts
            if isinstance(part, dict)
        )

        if not reply.strip():
            raise HTTPException(
                status_code=502,
                detail="AI provider returned an empty response"
            )

        return ChatResponse(
            reply=reply,
            model=GEMINI_MODEL
        )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="AI provider timed out"
        )

    except httpx.HTTPError:
        raise HTTPException(
            status_code=502,
            detail="Could not reach AI provider"
        )
