import json
import re

def extract_curly_bracket_content(text: str) -> str:
    """Extracts the curly bracketed json content within a string."""
    pattern = r"\{.*\}"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(0) if match else None

def parse_reasoning_response(response: str) -> str:
    """Extracts json formatting from a reasoning response."""
    try:
        data = json.loads(extract_curly_bracket_content(response))
        return data
    except (json.JSONDecodeError, TypeError):
        return ""
    
