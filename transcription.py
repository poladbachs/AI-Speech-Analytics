import whisper

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file to text using OpenAI's Whisper model.
    :param audio_path: Path to the audio file.
    :return: Transcribed text.
    """
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

if __name__ == "__main__":
    sample_text = transcribe_audio("sample_audio.wav")
    print("Transcribed Text:\n", sample_text)