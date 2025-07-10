from transformers import pipeline
import re

def redact_text(text, model_path="models/pii_ner_model"):
    ner_pipe = pipeline("ner", model=model_path, tokenizer=model_path, aggregation_strategy="simple")
    entities = ner_pipe(text)
    redacted_text = text
    for ent in sorted(entities, key=lambda e: e['start'], reverse=True):
        redacted_text = redacted_text[:ent['start']] + "[REDACTED]" + redacted_text[ent['end']:]
    return redacted_text

if __name__ == "__main__":
    test_text = "My name is Sarah Connor and my email is sarah@example.com."
    print("Original:", test_text)
    print("Redacted:", redact_text(test_text))
