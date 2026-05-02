import re
import unicodedata
from typing import List

from .email_list import whitelist_words, whitelist_patterns


TOKEN_RE = re.compile(r"\S+")

LEET_MAP = str.maketrans({
    "@": "a",
    "4": "a",
    "$": "s",
    "5": "s",
    "0": "o",
    "1": "i",
    "!": "i",
    "3": "e",
    "7": "t",
    "+": "t",
    "8": "b",
    "6": "g",
    "9": "g",
})

SUSPICIOUS_KEYWORDS = [
    
]
SYMBOL_SEPARATORS = ["-", "_", ".", ",", ":", ";", "/", "\\", "|", "~", "^"]

SUSPICIOUS_SYMBOLS = [
    "@",
    "$",
    "%",
    ">",
    "<",
    "/",
    "\\",
    "|",
    ":",
    ";",
    "=",
    "!",
    "#",
    "*",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "~",
    "^",
    "+",
]


def clean_token(token: str) -> str:
    t = token.strip()
    t = t.lstrip('(<[{\'"')
    t = t.rstrip('.,!?;:)]}\'"')
    return t


def normalize_token(token: str) -> str:
    t = unicodedata.normalize("NFKC", token.lower().strip())
    t = t.translate(LEET_MAP)

    for sep in SYMBOL_SEPARATORS:
        t = t.replace(sep, "")

    return t


def has_weird_unicode(token: str) -> bool:
    for ch in token:
        if ord(ch) > 127:
            cat = unicodedata.category(ch)
            if cat.startswith("L") or cat.startswith("N") or cat.startswith("S"):
                return True

    return False


def has_mixed_script_like_pattern(token: str) -> bool:
    alpha = sum(ch.isalpha() for ch in token)
    digit = sum(ch.isdigit() for ch in token)
    symbol = sum(not ch.isalnum() and not ch.isspace() for ch in token)

    return (alpha > 0 and digit > 0) or (alpha > 0 and symbol > 0)


def looks_like_split_keyword(token: str) -> bool:
    n = normalize_token(token)
    return any(keyword in n for keyword in SUSPICIOUS_KEYWORDS)


def has_repeated_separators(token: str) -> bool:
    patterns = ["..", "__", "--", "::", "//", "\\\\", "==", "~~", "^^"]
    return any(p in token for p in patterns)


def looks_like_obfuscated_domain(token: str) -> bool:
    t = token.lower()

    if "http" in t or "www." in t:
        return True

    if "." in t and any(ch.isalpha() for ch in t):
        return True

    return False


def looks_like_obfuscated_email(token: str) -> bool:
    t = token.lower()
    return "@" in t and "." in t


def is_whitelisted(token: str) -> bool:
    raw_token = token.strip()
    cleaned_token = clean_token(raw_token)

    raw_lower = raw_token.lower()
    cleaned_lower = cleaned_token.lower()

    if raw_lower in whitelist_words or cleaned_lower in whitelist_words:
        return True

    candidates = [raw_token, cleaned_token]

    for pattern in whitelist_patterns:
        try:
            for candidate in candidates:
                if re.fullmatch(pattern, candidate):
                    return True
        except re.error:
            continue

    return False


def extract_suspicious_tokens(email_text: str) -> List[str]:
    tokens = TOKEN_RE.findall(email_text or "")
    suspicious = []

    for tok in tokens:
        cleaned_tok = clean_token(tok)

        if not cleaned_tok:
            continue

        if is_whitelisted(cleaned_tok):
            continue

        token_lower = cleaned_tok.lower()

        has_digit = any(ch.isdigit() for ch in cleaned_tok)

        has_symbol = any(
            ch in cleaned_tok
            for ch in [
                "@",
                "$",
                "_",
                "%",
                ">",
                "<",
                "\\",
                "|",
                "=",
                "#",
                "*",
                "~",
                "^",
                "+",
            ]
        )

        has_url = (
            token_lower.startswith("http://")
            or token_lower.startswith("https://")
            or token_lower.startswith("www.")
        )

        has_email = "@" in cleaned_tok and "." in cleaned_tok

        alpha_count = sum(ch.isalpha() for ch in cleaned_tok)
        digit_count = sum(ch.isdigit() for ch in cleaned_tok)
        symbol_count = sum((not ch.isalnum() and not ch.isspace()) for ch in cleaned_tok)

        has_mixed_alpha_digit = alpha_count > 0 and digit_count > 0
        has_mixed_alpha_symbol = alpha_count > 0 and symbol_count > 0

        if (
            has_url
            or has_email
            or has_mixed_alpha_digit
            or has_mixed_alpha_symbol
            or (has_digit and len(cleaned_tok) > 3)
            or has_symbol
        ):
            suspicious.append(cleaned_tok)

    return suspicious