"""Curated, inspectable guidance retrieval; no network calls at analysis time."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GUIDANCE = [
    {
        "id": "FTC-PHISH-01",
        "title": "FTC: How to Recognize and Avoid Phishing Scams",
        "url": "https://consumer.ftc.gov/articles/how-recognize-avoid-phishing-scams",
        "text": (
            "Unexpected links, attachments, password requests, and payment demands can be phishing. "
            "Verify the request with a phone number or website you already know, not details in the message."
        ),
    },
    {
        "id": "FTC-RESPONSE-02",
        "title": "FTC: What to Do if You Suspect a Phishing Attack",
        "url": "https://consumer.ftc.gov/articles/how-recognize-avoid-phishing-scams",
        "text": (
            "Report suspicious messages. If you clicked a link or opened an attachment, "
            "update security software and scan the device. Use multifactor authentication."
        ),
    },
    {
        "id": "NIST-IR-03",
        "title": "NIST SP 800-61 Rev. 3: Incident Response",
        "url": "https://csrc.nist.gov/pubs/sp/800/61/r3/final",
        "text": (
            "Organizations should prepare for incident response, reduce incident impact, "
            "and improve detection, response, and recovery activities. Follow the "
            "organization's approved response process when compromise is suspected."
        ),
    },
    {
        "id": "CISA-PHISH-04",
        "title": "CISA: Recognize and Report Phishing",
        "url": "https://www.cisa.gov/secure-our-world/recognize-and-report-phishing",
        "text": (
            "Urgent requests for credentials or sensitive information, unexpected attachments, "
            "and suspicious links warrant independent verification and reporting."
        ),
    },
]


def retrieve_guidance(query: str, top_k: int = 3) -> list[dict]:
    """Return scored source snippets from the fixed local guidance set."""
    if not 1 <= top_k <= len(GUIDANCE):
        raise ValueError("top_k must be between 1 and the number of guidance entries")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    corpus = vectorizer.fit_transform([item["text"] for item in GUIDANCE])
    scores = cosine_similarity(vectorizer.transform([query]), corpus).ravel()
    ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return [
        {**GUIDANCE[index], "similarity": round(float(scores[index]), 4)}
        for index in ranked[:top_k]
        if scores[index] > 0
    ]
