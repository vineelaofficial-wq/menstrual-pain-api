from fastapi import FastAPI, Request
from pain_inference import predict_pain_cause_with_validation

app = FastAPI()

@app.post("/infer-pain-cause")
async def infer(request: Request):
    data = await request.json()
    result = predict_pain_cause_with_validation(data)
    return {
        "predicted_cause": result.predicted_cause.value,
        "confidence_level": result.confidence_level.value,
        "clinical_notes": result.clinical_notes,
        "red_flags": result.red_flags,
        "scores": {k.value: v for k, v in result.scores.items()}
    }
