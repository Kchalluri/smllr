# src/glove_loader.py
import os


def load_glove_model(glove_file_path="data/glove.6B.100d.txt"):
    print("🔄 Loading GloVe vectors...")
    glove_model = {}
    with open(glove_file_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = list(map(float, parts[1:]))
            glove_model[word] = vector
    print(f"Loaded {len(glove_model)} word vectors.")
    return glove_model


