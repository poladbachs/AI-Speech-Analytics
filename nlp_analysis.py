import spacy
import nltk
from nltk.tokenize import sent_tokenize
from textblob import TextBlob

nlp = spacy.load("en_core_web_sm")

nltk.download('punkt_tab')

def extract_keywords(text: str) -> list:
    """
    Extract keywords from the text using SpaCy.
    :param text: Input text.
    :return: List of keywords.
    """
    doc = nlp(text)
    keywords = [token.text for token in doc if token.is_stop is False and token.is_punct is False]
    return keywords

def perform_sentiment_analysis(text: str) -> dict:
    """
    Analyze sentiment of the text using TextBlob.
    :param text: Input text.
    :return: Dictionary with sentiment polarity and subjectivity.
    """
    blob = TextBlob(text)
    return {"polarity": blob.sentiment.polarity, "subjectivity": blob.sentiment.subjectivity}

def extract_named_entities(text: str) -> list:
    """
    Extract named entities using SpaCy.
    :param text: Input text.
    :return: List of named entities.
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def generate_summary(text: str, sentence_count: int = 3) -> str:
    """
    Generate a simple extractive summary by selecting the first few sentences.
    :param text: Input text.
    :param sentence_count: Number of sentences to return.
    :return: Summary text.
    """
    sentences = sent_tokenize(text)
    summary = " ".join(sentences[:sentence_count])
    return summary

if __name__ == "__main__":
    sample_text = "Your transcribed text goes here."
    print("Keywords:", extract_keywords(sample_text))
    print("Sentiment:", perform_sentiment_analysis(sample_text))
    print("Entities:", extract_named_entities(sample_text))
    print("Summary:", generate_summary(sample_text))