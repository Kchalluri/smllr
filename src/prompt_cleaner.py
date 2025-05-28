import os
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_prompt(prompt: str) -> str:
    prompt = prompt.lower()
    prompt = re.sub(r'\s+', ' ', prompt)
    prompt = re.sub(r'\b(?:just|actually|really|very|please|kindly|like|so|uh|um|sort of|kind of|maybe)\b', '', prompt)
    words = [w for w in prompt.split() if w not in stop_words]
    return ' '.join(words)

def build_optimized_prompt(pif: dict) -> str:
    topic = ", ".join(pif["entities"][:3])  # take top 3 for brevity
    focus = ", ".join(pif["actions"][:2])
    return f"{focus.capitalize()} the topic of {topic}. Keep it concise. Format with bullet points."


