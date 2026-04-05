import os
import logging
from typing import Dict, Optional
from dotenv import load_dotenv

try:
    from langchain_groq import ChatGroq
except Exception:  # pragma: no cover
    ChatGroq = None

load_dotenv()
logger = logging.getLogger(__name__)


def init_groq():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        logger.warning("GROQ_API_KEY not found. Falling back to rule-based recommendations.")
        return None

    if ChatGroq is None:
        logger.warning("langchain_groq not available. Falling back to rule-based recommendations.")
        return None

    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
        )
        logger.info("Groq LLM initialized")
        return llm
    except Exception as e:
        logger.error(f"Groq initialization failed: {e}")
        return None


class ClinicalRecommendationEngine:
    def __init__(self):
        self.llm = init_groq()

    def generate_recommendations(
        self,
        patient_age: int,
        predictions: Dict,
        clinical_notes: Optional[str] = None,
    ) -> Dict:
        if self.llm is None:
            return self._default_recommendations(predictions)

        try:
            summary = []
            for disease, data in predictions.items():
                summary.append(
                    f"{disease.replace('_', ' ').title()} -> "
                    f"{data['risk_category']} ({data['risk_score']:.2f})"
                )

            summary_text = "\n".join(summary)

            prompt = f"""
You are an expert clinical decision support AI.

Patient Age: {patient_age}

Predicted Risks:
{summary_text}

Instructions:
- Be medically accurate and structured
- Do NOT hallucinate diseases or facts
- Base reasoning ONLY on given risks and general clinical knowledge
- Be concise but informative
- Use bullet points where appropriate

Generate output STRICTLY in the following format:

### Prediction Summary
- Risk Score: <overall risk score if inferable>
- Risk Level: <LOW / MODERATE / HIGH / CRITICAL>

### Key Drivers
- Identify likely important clinical factors (e.g., age, hemoglobin, vitals)
- Mention whether they increase or decrease risk
- Label as High Impact / Moderate / Low Impact

### Clinical Insight
- Explain WHY key drivers influence the disease risk
- Distinguish between causal vs associated factors

### Recommended Changes
- Suggest targeted clinical or physiological improvements
- Focus on measurable factors (labs, vitals, behavior)

### Minimal Change Plan
- Provide small, realistic actions for short-term improvement

### Expected Outcome
- Briefly estimate how risk may change (qualitative or approximate range)

### Low Impact Factors
- Mention factors with minimal influence on risk

### Suggested Monitoring
- List tests, vitals, or follow-ups needed

### Lifestyle Recommendations
- Diet suggestions
- Physical activity guidance
- Sleep and stress advice

### Red Flags (Seek urgent care if)
- List critical warning signs

Important:
- Keep the tone clinical but understandable
- Do not over-exaggerate risk
- If data is limited, say \"based on available data\"
"""

            response = self.llm.invoke(prompt)

            high_risk = [
                disease.replace("_", " ").title()
                for disease, value in predictions.items()
                if value.get("risk_score", 0) >= 0.7
            ]

            return {
                "success": True,
                "recommendations": response.content,
                "model": "llama-3.1-8b-instant (Groq)",
                "high_risk_diseases": high_risk,
            }
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            return self._default_recommendations(predictions)

    def _default_recommendations(self, predictions: Dict) -> Dict:
        high = [d for d, v in predictions.items() if v.get("risk_score", 0) > 0.7]
        moderate = [d for d, v in predictions.items() if 0.4 < v.get("risk_score", 0) <= 0.7]

        text = "### Clinical Recommendations\n\n"

        if high:
            text += "High Risk Conditions:\n"
            for d in high:
                text += f"- {d.replace('_', ' ').title()} -> Immediate evaluation required\n"
            text += "\n"

        if moderate:
            text += "Moderate Risk Conditions:\n"
            for d in moderate:
                text += f"- {d.replace('_', ' ').title()} -> Monitor closely\n"
            text += "\n"

        text += (
            "### General Actions\n"
            "- Perform relevant diagnostic tests\n"
            "- Monitor vitals regularly\n"
            "- Follow clinical guidelines\n\n"
            "### Seek urgent care if\n"
            "- Sudden worsening of symptoms\n"
            "- Chest pain, breathlessness, confusion\n\n"
            "### Lifestyle Advice\n"
            "- Balanced diet\n"
            "- Hydration\n"
            "- Regular checkups\n"
        )

        return {
            "success": True,
            "recommendations": text,
            "model": "rule-based-fallback",
            "high_risk_diseases": [d.replace("_", " ").title() for d in high],
        }


_engine = None


def get_recommendation_engine():
    global _engine
    if _engine is None:
        _engine = ClinicalRecommendationEngine()
    return _engine
