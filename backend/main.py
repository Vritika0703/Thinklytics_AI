import json
from datetime import datetime
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .ai_engine import AIEngine, DecisionAnalysis

app = FastAPI(title="Decision Assistant AI API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = AIEngine()

# ─── In-memory history store ──────────────────────────────────────────────────
# For a real product, swap this for SQLite / PostgreSQL
decision_history: List[dict] = []


class DecisionRequest(BaseModel):
    situation: str


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "online", "version": "2.0", "message": "Decision Assistant AI is running."}


@app.get("/status")
async def status():
    """Returns current AI provider info for the frontend banner."""
    return {
        "live_mode": engine.is_live_mode,
        "provider": engine.provider or "demo",
        "message": (
            f"Live AI powered by {engine.provider.title()}" if engine.is_live_mode
            else "Demo mode — add a GEMINI_API_KEY to enable real AI analysis"
        )
    }

@app.post("/analyze", response_model=DecisionAnalysis)
async def analyze_decision(request: DecisionRequest):
    situation = request.situation.strip()

    # Validate minimum length — return soft warning instead of HTTP error
    if len(situation) < 5:
        return DecisionAnalysis(
            situation=situation,
            is_clear=False,
            clarity_message="Your input is too short. Try something like: 'Should I switch jobs?'",
            pros=[], cons=[], risks=[],
            emotional_score=0.0, logical_score=0.0, confidence_score=0.0,
            reasoning_summary="", practical_perspective="",
            optimistic_perspective="", worst_case_perspective="",
            structured_recommendation=""
        )

    try:
        analysis = engine.analyze_decision(situation)

        if analysis.is_clear:
            record = analysis.dict() if hasattr(analysis, 'dict') else analysis.model_dump()
            record.update({
                "id": len(decision_history) + 1,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "situation": situation
            })
            decision_history.append(record)
            # Keep last 50 decisions only
            if len(decision_history) > 50:
                decision_history.pop(0)

        return analysis

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
async def get_history():
    """Return decision history in reverse-chronological order."""
    return {"count": len(decision_history), "decisions": list(reversed(decision_history))}


@app.delete("/history")
async def clear_history():
    """Clear all stored decision history."""
    decision_history.clear()
    return {"message": "History cleared."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
