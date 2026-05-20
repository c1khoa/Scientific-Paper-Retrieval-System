import re

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text for lexical retrieval."""
    return TOKEN_PATTERN.findall(str(text).lower())