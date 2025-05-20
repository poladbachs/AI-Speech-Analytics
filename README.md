# 🏆 Spitch AI Speech Analytics System

## 🚀 Overview
**Spitch AI Speech Analytics System** is a practical demo project tailored for Spitch AI. It processes call audio data to provide actionable insights through:

- **Accurate Transcription:** Converts uploaded audio files into text.
- **Extractive Summarization:** Uses a controlled TextRank algorithm to extract key sentences directly from the transcription.
- **Sentiment Analysis & Entity Extraction:** Analyzes sentiment and extracts keywords and named entities to offer a quick overview of call dynamics.

This project demonstrates end-to-end capabilities—from speech-to-text conversion to insightful data extraction—using state-of-the-art NLP techniques.

---

## 📺 Demo
![Image](https://github.com/user-attachments/assets/dec85286-d8d5-4bc6-b24f-f606cd725422)

---

## 🔥 Key Features
- **Accurate Transcription:** Converts audio files (WAV, MP3, M4A) into text.
- **Extractive Summarization:** Uses TextRank to produce concise, faithful summaries without hallucinations.
- **Sentiment & Entity Analysis:** Provides overall tone, polarity, subjectivity, and identifies key entities.
- **Punctuation Restoration:** Enhances raw transcriptions by restoring punctuation for better readability.
- **Interactive Dashboard:** Built with Streamlit for an intuitive, real-time analysis interface.

---

## 🛠️ Tech Stack

| Technology    | Purpose                                                        |
|---------------|----------------------------------------------------------------|
| **Python**    | Core programming language                                      |
| **Spacy**     | NLP for named entity recognition and keyword extraction        |
| **NLTK**      | Tokenization and sentence splitting                             |
| **TextBlob**  | Sentiment analysis                                             |
| **Transformers** | Punctuation restoration and summarization using pre-trained models |
| **Sumy**      | Extractive summarization (TextRank)                             |
| **Streamlit** | Building an interactive web dashboard                           |

---

## 📌 How It Works
1. **Audio Processing:**  
   - Upload an audio file via the Streamlit dashboard.
   - The file is transcribed into text using a speech-to-text engine.

2. **Data Enrichment:**  
   - **Punctuation Restoration:** The raw transcription is enhanced to improve readability.
   - **NLP Analysis:** Extracts keywords, performs sentiment analysis, and identifies named entities.

3. **Extractive Summarization:**  
   - Applies TextRank (via Sumy) to select key sentences from the restored transcription.
   - Generates a concise summary that faithfully reflects the call's content.

4. **Interactive Insights:**  
   - The dashboard displays transcription, key insights (keywords, sentiment, entities), and the extractive summary in real time.

---

## 🔧 Setup & Installation

```bash
# Clone the repository
git clone https://github.com/YourUsername/spitch-ai-speech-analytics.git
cd spitch-ai-speech-analytics

# Install dependencies
pip install -r requirements.txt

# Then download the Spacy English model
python -m spacy download en_core_web_sm

# Run the Streamlit UI
streamlit run streamlit_app.py
```
