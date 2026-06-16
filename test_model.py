import sys
import os

# Add app directory to path to import clean_text
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app import clean_text

def test():
    try:
        from transformers import pipeline
    except ImportError:
        print("Transformers not installed")
        return

    model_path = os.path.join(os.path.dirname(__file__), 'distilbert_v2_indian_model')
    print(f"Loading from {model_path}")
    pipe = pipeline("text-classification", model=model_path, tokenizer=model_path)
    
    samples = [
        "The earth is flat and scientists are hiding the truth from us.",
        "Breaking: Aliens landed in New York today and spoke to the president.",
        "Apple announced a new iPhone model today with a faster processor and better camera.",
        "The stock market saw a slight increase following the Federal Reserve's decision to hold interest rates steady."
    ]

    for s in samples:
        c = clean_text(s)
        out = pipe(c, truncation=True, max_length=512)
        print(f"Text: {s[:50]}... -> {out}")

if __name__ == '__main__':
    test()
