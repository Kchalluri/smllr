import os
import spacy
from src.prompt_cleaner import clean_prompt

nlp = spacy.load("en_core_web_sm")

def extract_semantic_features(prompt: str) -> dict:
    doc = nlp(prompt)
    entities = [ent.text for ent in doc.ents]
    actions = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
    style = adjectives[0] if adjectives else "neutral"

    return {
        "entities": list(set(entities + nouns)),
        "actions": list(set(actions)),
        "constraints": adjectives,
        "style": style
    }

import numpy as np  # Make sure this is imported at the top

def get_prompt_embedding(prompt, glove, max_words=20):
    words = prompt.lower().split()[:max_words]
    vectors = [glove[word] for word in words if word in glove]
    if not vectors:
        return [0.0] * 100  # or 300 depending on GloVe dimensionality
    avg = np.mean(vectors, axis=0)
    return avg.tolist()



def generate_pif(prompt, glove, intent="auto", output="text") -> dict:
    features = extract_semantic_features(prompt)
    embedding = get_prompt_embedding(prompt, glove)

    return {
        "intent": intent,
        "entities": features["entities"],
        "actions": features["actions"],
        "constraints": features["constraints"],
        "style": features["style"],
        "embedding": embedding,
        "output": output
    }
