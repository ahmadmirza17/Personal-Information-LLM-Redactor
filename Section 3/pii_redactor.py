from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

class PIIRedactor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def redact_pii(self, text):
        ner_results = self.nlp(text)
        redacted_text = text
        for result in reversed(ner_results):
            entity = result['entity']
            if entity in ['B-PER', 'I-PER']:
                redacted_text = redacted_text[:result['start']] + '[NAME]' + redacted_text[result['end']:]
            elif entity in ['B-LOC', 'I-LOC']:
                redacted_text = redacted_text[:result['start']] + '[LOCATION]' + redacted_text[result['end']:]
            
        # Add simple regex for email and phone number
        import re
        redacted_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', redacted_text)
        redacted_text = re.sub(r'\+?[0-9]{8,}', '[PHONE-NUM]', redacted_text)
        
        # Simple NRIC redaction (this is a very basic implementation and might need refinement)
        redacted_text = re.sub(r'[STFG]\d{7}[A-Z]', '[NRIC]', redacted_text)
        
        return redacted_text

    def detect_pii(self, text):
        redacted = self.redact_pii(text)
        return {
            'name': '[NAME]' in redacted,
            'email': '[EMAIL]' in redacted,
            'phone': '[PHONE-NUM]' in redacted,
            'nric': '[NRIC]' in redacted,
            'location': '[LOCATION]' in redacted
        }