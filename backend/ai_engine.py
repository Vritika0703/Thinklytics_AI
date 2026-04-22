import os
import json
import re
import joblib
import re
from typing import List, Optional
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

load_dotenv()

# ─── Pydantic Schema (Strict) ────────────────────────────────────────────────

class DecisionAnalysis(BaseModel):
    situation: str
    is_clear: bool = True                # False if input is too vague
    clarity_message: Optional[str] = None  # e.g. "Please add more details"
    pros: List[dict]
    cons: List[dict]
    risks: List[dict]
    emotional_score: float               # 0.0 – 1.0
    logical_score: float                 # 0.0 – 1.0
    confidence_score: float              # 0.0 – 1.0 (LLM self-assessed)
    reasoning_summary: str               # 2–3 line final synthesis
    practical_perspective: str
    optimistic_perspective: str
    worst_case_perspective: str
    structured_recommendation: str
    ml_category: Optional[str] = None
    ml_risk: Optional[str] = None
    ml_sentiment: Optional[str] = None

    @field_validator("confidence_score", "emotional_score", "logical_score", mode="before")
    @classmethod
    def clamp_float(cls, v):
        """Ensure scores are always in [0.0, 1.0], no matter what the LLM returns."""
        return max(0.0, min(1.0, float(v)))


# ─── Validation Helpers ───────────────────────────────────────────────────────

def _is_input_clear(situation: str) -> tuple[bool, Optional[str]]:
    """
    Heuristic check for vague / emotional rants before hitting the LLM.
    Returns (is_clear, message).
    """
    cleaned = situation.strip()

    if len(cleaned) < 15:
        return False, "Your input is very short. Could you describe your situation in a bit more detail?"

    # Check for pure emotional venting – very few real words, lots of punctuation / caps
    word_count = len(re.findall(r'\b[a-zA-Z]{2,}\b', cleaned))
    if word_count < 4:
        return False, "This looks more like a feeling than a decision. Try framing it as a question, e.g. 'Should I…?'"

    # Check for a recognisable decision structure
    decision_signals = ["should", "switch", "leave", "join", "move", "quit", "start", "stop",
                        "take", "accept", "reject", "stay", "go", "choose", "decide", "break up",
                        "change", "learn", "buy", "sell", "invest", "apply"]
    lower = cleaned.lower()
    has_signal = any(s in lower for s in decision_signals)
    if not has_signal and word_count < 8:
        return False, "I need a bit more clarity. What are you deciding between? E.g. 'Should I leave my current job for a startup?'"

    return True, None


# ─── AI Engine ────────────────────────────────────────────────────────────────

# ─── Shared Prompt ───────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """You are a professional strategic decision analyst.

Analyze the following situation and return a JSON object with EXACTLY these keys:
- pros: list of 3-4 objects, each with:
    "text": a specific, situation-aware advantage (string)
    "weight": importance 0.0-1.0 (float)
- cons: list of 3 objects, each with:
    "text": a specific, situation-aware disadvantage (string)
    "weight": importance 0.0-1.0 (float)
- risks: list of 3 objects, each with:
    "text": a specific risk tied to THIS situation (string)
    "severity": one of "low", "medium", "high"
- emotional_score: float 0.0-1.0 calibrated as follows:
    0.0-0.25 = pure logic (e.g. choosing a software library, optimizing a budget spreadsheet)
    0.25-0.50 = mostly logical with minor emotional component (e.g. career skill choice, investing small savings)
    0.50-0.75 = mixed — significant emotional AND logical factors (e.g. job switch, moving cities)
    0.75-1.0 = highly emotional — relationships, life-altering choices, grief, fear (e.g. breakup, quitting everything)
    Use the full range — do NOT default to High for every decision.
- logical_score: float 0.0-1.0 calibrated as follows:
    0.0-0.35 = very ambiguous, hard to reason logically (e.g. "should I be happy?")
    0.35-0.65 = moderate logical clarity — some data available but uncertain (e.g. career switches)
    0.65-1.0 = strong logical case — data, comparables, clear trade-offs available (e.g. salary negotiation, financial decisions)
- confidence_score: float 0.0-1.0 — your confidence given the info provided. Be honest; lower if vague or missing context.
- reasoning_summary: 2-3 sentence synthesis naming the SPECIFIC trade-off in THIS situation (not generic)
- practical_perspective: 2-3 sentences of concrete, actionable logistical advice specific to THIS situation
- optimistic_perspective: 2-3 sentences from a growth-oriented viewpoint specific to THIS situation
- worst_case_perspective: 2-3 sentences from a cautious risk-averse viewpoint specific to THIS situation
- structured_recommendation: 1 clear, actionable sentence tailored to THIS situation

CRITICAL:
- Every piece of text MUST reference specifics from the situation — do NOT give generic advice
- reasoning_summary must name the actual trade-off (e.g. "salary vs equity", "security vs freedom", "love vs incompatibility")
- emotional_score and logical_score MUST reflect the actual nature of this decision — vary them realistically
- Return ONLY valid JSON, no markdown, no explanation

Situation to analyze: "{situation}"
"""


# ─── AI Engine ────────────────────────────────────────────────────────────────

class AIEngine:
    def __init__(self):
        # Priority: Gemini (free tier) → OpenAI → mock
        self.gemini_key  = os.getenv("GEMINI_API_KEY")
        self.openai_key  = os.getenv("OPENAI_API_KEY")
        self.provider    = None
        self._gemini_model = None
        self._openai_client = None

        # Load Local ML Models
        try:
            base_dir = os.path.dirname(__file__)
            self.ml_category = joblib.load(os.path.join(base_dir, 'ml', 'model_category.pkl'))
            self.ml_risk = joblib.load(os.path.join(base_dir, 'ml', 'model_risk.pkl'))
            self.ml_sentiment = joblib.load(os.path.join(base_dir, 'ml', 'model_sentiment.pkl'))
            print("✅ Local Scikit-Learn Models loaded successfully!")
        except Exception as e:
            print(f"⚠️  ML Models not found: {e}")
            self.ml_category = None
            self.ml_risk = None
            self.ml_sentiment = None

        if self.gemini_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=self.gemini_key)
                # Try models in order — each has its own separate free-tier quota
                self._gemini_model_name = None
                for candidate in [
                    "gemini-2.5-flash-lite",
                    "gemini-2.5-flash",
                    "gemini-2.0-flash-lite",
                    "gemini-2.0-flash",
                ]:
                    self._gemini_model_name = candidate
                    print(f"✅ AI Engine: Using Gemini ({candidate})")
                    break
                self.provider = "gemini"
            except Exception as e:
                print(f"⚠️  Gemini init failed: {e}")

        if not self.provider and self.openai_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.openai_key)
                self.provider = "openai"
                print("✅ AI Engine: Using OpenAI")
            except Exception as e:
                print(f"⚠️  OpenAI init failed: {e}")

        if not self.provider:
            print("⚠️  AI Engine: No API key found — running in demo mode")

    @property
    def is_live_mode(self) -> bool:
        return self.provider is not None

    def analyze_decision(self, situation: str) -> DecisionAnalysis:
        is_clear, clarity_message = _is_input_clear(situation)
        if not is_clear:
            return self._unclear_response(situation, clarity_message)

        ml_cat, ml_risk, ml_sent = None, None, None
        ml_context = ""
        if self.ml_category and self.ml_risk and self.ml_sentiment:
            ml_cat = self.ml_category.predict([situation])[0]
            ml_risk = self.ml_risk.predict([situation])[0]
            ml_sent = self.ml_sentiment.predict([situation])[0]
            ml_context = f"\n\nML Metadata (Local Classified): Category={ml_cat}, Risk={ml_risk}, Sentiment={ml_sent}. Align analysis with these baseline tags."

        try:
            if self.provider == "gemini":
                result = self._call_gemini(situation, ml_context)
            elif self.provider == "openai":
                result = self._call_openai(situation, ml_context)
            else:
                result = self._get_mock_analysis(situation)
        except Exception as e:
            print(f"AI call failed ({self.provider}): {e} — falling back to mock")
            result = self._get_mock_analysis(situation)
            
        result.ml_category = ml_cat
        result.ml_risk = ml_risk
        result.ml_sentiment = ml_sent
        return result

    def _parse_llm_response(self, raw_text: str, situation: str) -> DecisionAnalysis:
        """Parse and validate JSON from any LLM provider."""
        # Strip markdown fences if present
        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw_text.strip(), flags=re.MULTILINE)
        data = json.loads(cleaned)
        data["situation"] = situation
        data["is_clear"] = True
        data["clarity_message"] = None
        data.setdefault("reasoning_summary", "Analysis complete.")
        data.setdefault("emotional_score", 0.5)
        data.setdefault("logical_score", 0.5)
        data.setdefault("confidence_score", 0.75)
        return DecisionAnalysis(**data)

    def _call_gemini(self, situation: str, ml_context: str = "") -> DecisionAnalysis:
        from google import genai
        from google.genai import types

        prompt = ANALYSIS_PROMPT.format(situation=situation) + ml_context
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.7,
            max_output_tokens=1024,
            http_options=types.HttpOptions(timeout=12000),
        )

        # Try each model in cascade — each has its own separate free-tier quota
        models_to_try = [
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash",
        ]

        last_error = None
        for model in models_to_try:
            try:
                response = self._gemini_client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                print(f"✅ Gemini live response via {model}")
                return self._parse_llm_response(response.text, situation)
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    print(f"⚠️  {model} quota exhausted — trying next model...")
                    last_error = e
                    continue
                else:
                    raise  # Re-raise non-quota errors immediately

        raise Exception(f"All Gemini models exhausted. Last error: {last_error}")

    def _call_openai(self, situation: str, ml_context: str = "") -> DecisionAnalysis:
        prompt = ANALYSIS_PROMPT.format(situation=situation) + ml_context
        response = self._openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a structured decision intelligence system. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return self._parse_llm_response(response.choices[0].message.content, situation)

    def _unclear_response(self, situation: str, message: str) -> DecisionAnalysis:
        """Return a graceful 'needs more clarity' response without hitting the LLM."""
        return DecisionAnalysis(
            situation=situation,
            is_clear=False,
            clarity_message=message,
            pros=[],
            cons=[],
            risks=[],
            emotional_score=0.0,
            logical_score=0.0,
            confidence_score=0.0,
            reasoning_summary="",
            practical_perspective="",
            optimistic_perspective="",
            worst_case_perspective="",
            structured_recommendation=""
        )

    def _get_mock_analysis(self, situation: str) -> DecisionAnalysis:
        """
        Situation-aware fallback when no API key is present.
        - Detects topic domain (career, relationship, learning, finance, location)
        - Returns domain-specific pros/cons/risks
        - Confidence correlates with logical score (no more contradictions)
        - Scores are deterministic per input via hash-seeded RNG
        """
        import hashlib, random

        text = situation.lower()
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 32)
        rng = random.Random(seed)

        # ── Score generation (correlated) ────────────────────────────────────
        logical   = round(rng.uniform(0.52, 0.93), 2)
        emotional = round(rng.uniform(0.18, 0.82), 2)

        # Confidence tracks logical closely, with a small ±0.05 jitter
        confidence = round(min(0.97, max(0.60, logical + rng.uniform(-0.05, 0.05))), 2)

        clarity      = "strong"   if logical   > 0.75 else ("moderate" if logical   > 0.55 else "limited")
        emo_label    = "high"     if emotional  > 0.62 else ("moderate" if emotional > 0.38 else "low")
        conf_label   = "high"     if confidence > 0.80 else ("moderate" if confidence > 0.65 else "low")

        # ── Domain detection ─────────────────────────────────────────────────
        career_kw      = ["job", "career", "startup", "company", "salary", "role", "work", "hire",
                          "offer", "intern", "quit", "resign", "switch", "promotion", "corporate"]
        relation_kw    = ["break up", "breakup", "relationship", "partner", "boyfriend", "girlfriend",
                          "marry", "marriage", "divorce", "date", "dating", "love", "together"]
        learning_kw    = ["learn", "study", "course", "degree", "skill", "bootcamp", "university",
                          "college", "major", "ml", "machine learning", "coding", "programming"]
        finance_kw     = ["invest", "stock", "crypto", "money", "loan", "debt", "savings", "buy",
                          "sell", "rent", "mortgage", "afford", "budget", "expense", "equity"]
        location_kw    = ["move", "relocate", "city", "country", "abroad", "travel", "nyc", "sf",
                          "new york", "san francisco", "london", "remote", "abroad"]

        def matches(keywords): return any(k in text for k in keywords)

        if matches(relation_kw):
            domain = "relationship"
        elif matches(learning_kw):
            domain = "learning"
        elif matches(finance_kw) and not matches(career_kw):
            domain = "finance"
        elif matches(location_kw):
            domain = "location"
        else:
            domain = "career"   # default

        # ── Domain content banks ─────────────────────────────────────────────
        CONTENT = {
            "career": {
                "pros": [
                    "Higher growth trajectory and skill acceleration than current role",
                    "Greater ownership and decision-making responsibility",
                    "Exposure to new technologies and industry networks",
                    "Potential for significant upside in compensation or equity",
                ],
                "cons": [
                    "Short-term income or job security may decrease",
                    "Leaving established team relationships and institutional knowledge",
                    "Longer ramp-up time in an unfamiliar environment",
                    "Higher performance pressure without a safety net",
                ],
                "risks": [
                    ("New role may not deliver on stated expectations", "high"),
                    ("Company culture may be misaligned with your working style", "medium"),
                    ("Industry volatility could affect the role's stability", "medium"),
                ],
                "practical": "Validate the opportunity thoroughly before committing: confirm role scope, team quality, and financial runway of the company. Request a 30/60/90-day plan and review the equity terms carefully.",
                "optimistic": "A well-timed career move can compound your earnings and seniority by years. If the fundamentals are solid, the upside of acting decisively far outweighs the discomfort of transition.",
                "worst_case": "If it doesn't work out, you re-enter the market with more experience and a stronger story. Maintain your network proactively during the transition and negotiate a reasonable notice period.",
                "verdict": "Proceed — but validate the fundamentals (funding, culture, role scope) before signing.",
            },
            "relationship": {
                "pros": [
                    "Removing a source of persistent emotional stress improves mental clarity",
                    "Freedom to invest energy in personal growth and new connections",
                    "Honest communication about incompatibilities builds long-term self-awareness",
                    "Allows both parties to pursue more compatible relationships",
                ],
                "cons": [
                    "Short-term emotional pain and loneliness are real and significant",
                    "Shared social circles and logistics may become complicated",
                    "Loss of intimacy and the comfort of an established bond",
                    "Uncertainty about what comes next can feel destabilising",
                ],
                "risks": [
                    ("Emotional decision made prematurely — issues may actually be solvable", "high"),
                    ("Idealising independence and underestimating the adjustment period", "medium"),
                    ("External pressure (friends/family) influencing the decision unfairly", "low"),
                ],
                "practical": "Before any irreversible step, determine whether you've had a direct, honest conversation about the core issues. If not, that is the first action — not the breakup. Clarity requires communication.",
                "optimistic": "People who make courageous decisions aligned with their values consistently report higher long-term life satisfaction. Temporary pain for lasting authenticity is worth it.",
                "worst_case": "Loneliness and regret are real risks. Ensure you have a support network in place and are not making this decision during a peak emotional moment. Give yourself time to reflect before committing.",
                "verdict": "Reflect deeply and communicate honestly before making an irreversible decision.",
            },
            "learning": {
                "pros": [
                    "Skill acquisition compounds over time — early investment pays disproportionately",
                    "Differentiates your profile in a competitive job market",
                    "Opens doors to higher-leverage career opportunities",
                    "Builds confidence and problem-solving capacity across domains",
                ],
                "cons": [
                    "Significant time commitment that competes with other priorities",
                    "Learning curve can be steep and progress feels slow initially",
                    "Risk of choosing a skill that becomes less relevant than expected",
                    "Financial cost of courses, tools, or lost income during study",
                ],
                "risks": [
                    ("Market demand for the skill may shift before you reach proficiency", "medium"),
                    ("Tutorial progress does not translate directly to real-world application", "medium"),
                    ("Motivation drop-off after the initial excitement phase", "high"),
                ],
                "practical": "Set a specific outcome: 'I will build X project in Y weeks.' Structured milestones prevent the common trap of indefinite learning without application. Use the 70/20/10 rule: 70% practice, 20% projects, 10% theory.",
                "optimistic": "Every technical skill you add creates compounding optionality. The discomfort of learning something genuinely hard is the exact signal that you're building real competitive advantage.",
                "worst_case": "If the skill turns out to be less valuable, the meta-skill of learning something hard transfers completely. You're never truly back to zero once you've gone deep on anything.",
                "verdict": "Commit to one path and execute with structured milestones — avoid the trap of switching before going deep.",
            },
            "finance": {
                "pros": [
                    "Potential for returns significantly above inflation and savings accounts",
                    "Builds financial discipline and understanding of capital allocation",
                    "Diversifies income beyond active employment",
                    "Early financial moves compound most powerfully over time",
                ],
                "cons": [
                    "Capital at risk — losses can be significant in volatile markets",
                    "Requires sustained attention and ongoing research",
                    "Emotional bias (FOMO, panic selling) often undermines returns",
                    "Liquidity may be locked up for extended periods",
                ],
                "risks": [
                    ("Market timing risk — entering at a peak inflates downside exposure", "high"),
                    ("Concentration risk if investing in a single asset or sector", "high"),
                    ("Tax implications not fully accounted for in return projections", "medium"),
                ],
                "practical": "Only deploy capital you can afford to lose entirely over the investment horizon. Define your exit conditions before you enter. Dollar-cost averaging reduces timing risk on volatile assets.",
                "optimistic": "Long-horizon investing in diversified, high-quality assets has historically rewarded patience. The difference between investing at 22 vs 32 is measured in decades of compounding.",
                "worst_case": "Maximum loss = capital deployed. Ensure this does not compromise your emergency fund (3–6 months expenses) or near-term obligations. Never invest borrowed money in volatile assets.",
                "verdict": "Proceed only with capital you can lock away for the full horizon without impacting your financial floor.",
            },
            "location": {
                "pros": [
                    "New environments dramatically accelerate personal growth and perspective",
                    "Access to better economic opportunities or quality of life improvements",
                    "Removes you from limiting local social dynamics or comfort zones",
                    "Adventure and novelty have documented positive effects on creativity",
                ],
                "cons": [
                    "Distance from family and established support networks is a real cost",
                    "Upfront relocation costs and the administrative burden of resettling",
                    "Social isolation during the integration period can be significant",
                    "Cost of living in destination may differ substantially from expectations",
                ],
                "risks": [
                    ("Opportunity that motivated the move may not materialise as expected", "high"),
                    ("Underestimating how long it takes to build a new social support system", "medium"),
                    ("Home costs or family circumstances may require return sooner than planned", "medium"),
                ],
                "practical": "Visit the destination for at least 2 weeks before committing. Map out your financial plan for the first 6 months including housing, transport, and emergency buffer. Have a re-evaluation checkpoint at 90 days.",
                "optimistic": "Relocation is one of the highest-leverage personal decisions. The people you meet and opportunities you access in a new environment often define the next decade of your trajectory.",
                "worst_case": "If it doesn't work, you move back — with life experience that cannot be gained any other way. The hidden cost of not moving (wondering 'what if') often exceeds the cost of a failed attempt.",
                "verdict": "Go — but plan the first 6 months financially before you book anything.",
            },
        }

        c = CONTENT[domain]

        # Pick weighted pros/cons (vary selection slightly per hash)
        pros_pool = c["pros"]
        cons_pool = c["cons"]
        rng2 = random.Random(seed + 1)   # different offset for list shuffles
        rng2.shuffle(pros_pool)
        rng2.shuffle(cons_pool)
        pros = [{"text": t, "weight": round(rng.uniform(0.6, 0.95), 2)} for t in pros_pool[:4]]
        cons = [{"text": t, "weight": round(rng.uniform(0.4, 0.80), 2)} for t in cons_pool[:3]]
        risks = [{"text": t, "severity": s} for t, s in c["risks"]]

        # ── Reasoning summary using actual scores ────────────────────────────
        reasoning = (
            f"The logical case for this decision is {clarity} ({int(logical*100)}% logical score). "
            f"Emotional investment is {emo_label} ({int(emotional*100)}%), which {'may introduce bias — try to separate feelings from facts' if emotional > 0.5 else 'is manageable and unlikely to distort analysis'}. "
            f"Overall confidence is {conf_label} at {int(confidence*100)}%."
        )

        return DecisionAnalysis(
            situation=situation,
            is_clear=True,
            clarity_message=None,
            pros=pros,
            cons=cons,
            risks=risks,
            emotional_score=emotional,
            logical_score=logical,
            confidence_score=confidence,
            reasoning_summary=reasoning,
            practical_perspective=c["practical"],
            optimistic_perspective=c["optimistic"],
            worst_case_perspective=c["worst_case"],
            structured_recommendation=c["verdict"],
        )

