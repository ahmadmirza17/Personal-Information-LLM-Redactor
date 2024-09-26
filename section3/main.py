from fastapi import FastAPI
from pydantic import BaseModel
from pii_redactor import PIIRedactor

app = FastAPI()
redactor = PIIRedactor()

class TextInput(BaseModel):
    text: str

@app.post("/redact")
async def redact_text(input: TextInput):
    redacted_text = redactor.redact_pii(input.text)
    pii_detected = redactor.detect_pii(input.text)
    return {"redacted_text": redacted_text, "pii_detected": pii_detected}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)