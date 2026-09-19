"""Tokenizer deliberately mirrored by src/model/tokenizer.ts."""
import re

TOKENIZER_VERSION = "lowercase-word-punctuation-v1"

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+|[.!?,]", text.lower())
