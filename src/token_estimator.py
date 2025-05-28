import os
import tiktoken

def count_tokens(prompt: str, model='gpt-4') -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(prompt))

def estimate_cost(tokens: int, model='gpt-4') -> float:
    price_per_1k = 0.03 if 'gpt-4' in model else 0.0015
    return round((tokens / 1000) * price_per_1k, 5)

def sustainability_estimate(tokens: int) -> str:
    watt_hours = tokens * 0.0005
    return f"{watt_hours:.4f} Wh"