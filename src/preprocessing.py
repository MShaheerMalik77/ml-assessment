import re
import pandas as pd

STOPWORDS = {
    "the", "is", "and", "a", "an", "to", "of", "in", "on", "for", "with", "that",
    "this", "it", "as", "at", "by", "from", "be", "are", "was", "were", "has",
    "had", "have", "he", "his", "she", "hers", "him", "her",
}

LABEL_MAP = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)