"""Deterministic, offline signals extracted from email text."""

import re
from urllib.parse import urlparse

URL_RE = re.compile(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+', re.IGNORECASE)
URGENCY_RE = re.compile(
    r"\b(urgent|immediately|final warning|suspended|expires?|within \d+ hours?)\b",
    re.IGNORECASE,
)
CREDENTIAL_RE = re.compile(
    r"\b(password|login|verify (?:your )?account|credential|sign[ -]?in|otp|"
    r"one[- ]time password)\b",
    re.IGNORECASE,
)
PAYMENT_RE = re.compile(
    r"\b(invoice|wire transfer|gift card|crypto|bitcoin|payment|bank account)\b",
    re.IGNORECASE,
)
ATTACHMENT_RE = re.compile(
    r"\b(attachment|attached|enable macros?|\.exe|\.zip|\.html)\b",
    re.IGNORECASE,
)
SHORTENER_HOSTS = frozenset({"bit.ly", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly"})


def _hits(pattern: re.Pattern[str], text: str, limit: int = 8) -> list[str]:
    return list(dict.fromkeys(match.group(0).lower() for match in pattern.finditer(text)))[:limit]


def forensic_indicators(email_text: str) -> dict:
    """Identify suspicious patterns without opening links or executing content."""
    urls = [url.rstrip(".,;!?)") for url in URL_RE.findall(email_text)[:20]]
    shortened = []
    for url in urls:
        candidate = url if url.lower().startswith("http") else "http://" + url
        host = urlparse(candidate).hostname or ""
        if host.lower() in SHORTENER_HOSTS:
            shortened.append(url)
    letters = sum(char.isalpha() for char in email_text)
    return {
        "url_count": len(urls),
        "urls_observed_do_not_visit": urls,
        "shortened_urls": shortened,
        "urgency_terms": _hits(URGENCY_RE, email_text),
        "credential_terms": _hits(CREDENTIAL_RE, email_text),
        "payment_terms": _hits(PAYMENT_RE, email_text),
        "attachment_terms": _hits(ATTACHMENT_RE, email_text),
        "excessive_exclamation": email_text.count("!") >= 4,
        "all_caps_ratio": round(sum(char.isupper() for char in email_text) / max(1, letters), 3),
    }
