# main.py
from transcription import transcribe_audio
from nlp_analysis import extract_keywords, perform_sentiment_analysis, extract_named_entities, generate_summary

def process_audio(audio_path: str):
    text = transcribe_audio(audio_path)
    
    analysis = {
        "transcription": text,
        "keywords": extract_keywords(text),
        "sentiment": perform_sentiment_analysis(text),
        "entities": extract_named_entities(text),
        "summary": generate_summary(text)
    }
    return analysis

if __name__ == "__main__":
    results = process_audio("sample_audio.wav")
    print("Transcription:\n", results["transcription"])
    print("\nKeywords:", results["keywords"])
    print("\nSentiment:", results["sentiment"])
    print("\nEntities:", results["entities"])
    print("\nSummary:", results["summary"])