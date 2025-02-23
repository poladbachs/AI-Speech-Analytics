import spacy
import nltk
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

nlp = spacy.load("en_core_web_sm")
nltk.download("punkt")

punct_tokenizer = AutoTokenizer.from_pretrained("felflare/bert-restore-punctuation")
punct_model = AutoModelForTokenClassification.from_pretrained("felflare/bert-restore-punctuation")

def extract_keywords(text: str) -> list:
    doc = nlp(text)
    return [token.text for token in doc if not token.is_stop and not token.is_punct]

def perform_sentiment_analysis(text: str) -> dict:
    blob = TextBlob(text)
    return {"polarity": blob.sentiment.polarity, "subjectivity": blob.sentiment.subjectivity}

def extract_named_entities(text: str) -> list:
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def clean_text(text: str) -> str:
    for token in ["[CLS]", "[SEP]", "[UNK]"]:
        text = text.replace(token, "")
    return text.strip()

def restore_punctuation(text: str) -> str:
    if not text.strip():
        return ""
    inputs = punct_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = punct_model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
    tokens = punct_tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    mapping = {"O": "", "COMMA": ",", "PERIOD": ".", "QUESTION": "?"}
    restored_tokens = []
    for token, pred in zip(tokens, predictions):
        token = token.replace("Ġ", "")
        punct = mapping.get(punct_model.config.id2label[pred], "")
        restored_tokens.append(token + punct)
    restored_text = punct_tokenizer.convert_tokens_to_string(restored_tokens)
    return clean_text(restored_text)

def extractive_summary(text: str, sentence_count: int = 3) -> str:
    if not text.strip():
        return ""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    extractor = TextRankSummarizer()
    summary_sentences = extractor(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary_sentences)

def generate_summary(text: str) -> str:
    restored_text = restore_punctuation(text)
    return extractive_summary(restored_text, sentence_count=3)

if __name__ == "__main__":
    sample_text = (
        "Growing up with two other siblings in the house, I can honestly say that we were raised pretty well and I never expected for Anything devastating to happen to any of us But a couple of years ago it was brought to my attention that one of my siblings Who was only two years younger than me has been suffering Through a drug addiction that has lasted over the course of a decade"
    )
    print("Keywords:", extract_keywords(sample_text))
    print("Sentiment:", perform_sentiment_analysis(sample_text))
    print("Entities:", extract_named_entities(sample_text))
    print("Extractive Summary:", extractive_summary(restore_punctuation(sample_text), sentence_count=3))
    print("Final Summary:", generate_summary(sample_text))
