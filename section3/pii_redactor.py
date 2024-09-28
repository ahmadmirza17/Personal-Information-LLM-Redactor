import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

class PIIRedactor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def redact_pii(self, text):
        # NER-based redaction
        ner_results = self.nlp(text)
        redacted_text = text
        for result in reversed(ner_results):
            entity = result['entity']
            if entity in ['B-PER', 'I-PER']:
                redacted_text = redacted_text[:result['start']] + '[NAME]' + redacted_text[result['end']:]
            elif entity in ['B-LOC', 'I-LOC']:
                redacted_text = redacted_text[:result['start']] + '[LOCATION]' + redacted_text[result['end']:]

        # Regex-based redaction
        # Email
        redacted_text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL]',
            redacted_text
        )
        
        # Phone Number Pattern
        phone_pattern = r'\b(?:\+?65\s?)?[689]\d{3}\s?\d{4}\b'
        redacted_text = re.sub(phone_pattern, '[PHONE-NUM]', redacted_text)
        
        # NRIC
        redacted_text = re.sub(
            r'\b[STFG]\d{7}[A-Z]\b',
            '[NRIC]',
            redacted_text
        )
        
        # Singapore address with postal code
        redacted_text = re.sub(
            r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Terrace|Ter|Place|Pl|Court|Ct)[,.\s]+(?:[A-Za-z]+[,.\s]+)*Singapore\s+\d{6}\b',
            '[ADDRESS]',
            redacted_text,
            flags=re.IGNORECASE
        )
        
        # Standalone Singapore postal code
        redacted_text = re.sub(
            r'\b\d{6}\b',
            '[POSTAL-CODE]',
            redacted_text
        )

        return redacted_text

    def detect_pii(self, text):
        redacted = self.redact_pii(text)
        return {
            'name': '[NAME]' in redacted,
            'email': '[EMAIL]' in redacted,
            'phone': '[PHONE-NUM]' in redacted,
            'nric': '[NRIC]' in redacted,
            'location': '[LOCATION]' in redacted or '[ADDRESS]' in redacted or '[POSTAL-CODE]' in redacted
        }
