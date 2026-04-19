"""
legal_validator.py
------------------
Validates whether:
  1. An uploaded PDF is a legal document.
  2. A user's chat query is legal in nature.

HOW IT WORKS:
- Uses a zero-shot keyword + heuristic approach (no extra API call needed).
- For stricter validation, it optionally calls Groq to classify the document.
- Returns (is_valid: bool, reason: str) tuples.

GROQ API KEY LOCATION IN THIS FILE:
  - Passed in via environment variable GROQ_API_KEY (set in .env)
  - Do NOT hardcode it here.
"""

import os
import re
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# LEGAL DOCUMENT KEYWORDS (heuristic layer)
# ─────────────────────────────────────────────
LEGAL_DOCUMENT_KEYWORDS = [
    # Contract / Agreement terms
    "agreement", "contract", "terms and conditions", "whereas", "hereby",
    "hereinafter", "party", "parties", "consideration", "obligations",
    "covenant", "indemnify", "indemnification", "liability", "warranties",
    "representations", "intellectual property", "confidentiality", "non-disclosure",
    "nda", "termination", "governing law", "jurisdiction", "arbitration",
    "dispute resolution", "force majeure", "amendment", "addendum",

    # Court / Litigation
    "plaintiff", "defendant", "petitioner", "respondent", "affidavit",
    "deposition", "subpoena", "motion", "injunction", "judgement", "judgment",
    "verdict", "court order", "summons", "complaint", "appeal", "brief",
    "exhibit", "counsel", "attorney", "solicitor", "barrister",

    # Legislation / Regulatory
    "act", "statute", "regulation", "section", "clause", "article",
    "provision", "pursuant", "statutory", "compliance", "regulatory",
    "ordinance", "bill", "legislature", "amendment", "enforcement",

    # Corporate / Business Law
    "incorporation", "articles of incorporation", "bylaws", "shareholder",
    "board of directors", "fiduciary", "due diligence", "merger", "acquisition",
    "memorandum", "prospectus", "securities", "equity", "dividend",

    # Property / Real Estate
    "deed", "title", "conveyance", "easement", "lien", "mortgage",
    "lease", "tenancy", "landlord", "tenant", "property rights",

    # Wills / Estates
    "will", "testament", "testator", "beneficiary", "executor",
    "probate", "estate", "trust", "trustee", "inheritance",

    # IP Law
    "patent", "trademark", "copyright", "infringement", "license",
    "royalty", "trade secret", "fair use",

    # General Legal Latin / Phrases
    "pursuant to", "notwithstanding", "in witness whereof",
    "signed and sealed", "executed as", "force of law",
    "legal notice", "legal document", "legal agreement",
]

# ─────────────────────────────────────────────
# NON-LEGAL CHAT QUERY PATTERNS
# ─────────────────────────────────────────────
NON_LEGAL_QUERY_PATTERNS = [
    # Casual / Off-topic
    r"\b(joke|funny|meme|movie|music|song|recipe|food|cook|weather|sport|game|play)\b",
    r"\b(how are you|what's up|hello|hi there|whats up|tell me about yourself)\b",
    r"\b(math|calculate|equation|algebra|programming|code|python|javascript)\b",
    r"\b(news|politics|celebrity|gossip|social media|instagram|tiktok)\b",
    # Harmful content
    r"\b(hack|exploit|illegal|fraud|scam|steal|kill|harm|weapon|drug|smuggle)\b",
    # Medical (unless malpractice, which has legal keywords)
    r"\b(diagnose|symptom|medicine|prescription|therapy|treatment|cure)\b",
]

LEGAL_QUERY_KEYWORDS = [
    "contract", "law", "legal", "clause", "liability", "rights", "court",
    "agreement", "statute", "regulation", "compliance", "damages", "penalty",
    "obligation", "jurisdiction", "attorney", "counsel", "plaintiff", "defendant",
    "dispute", "breach", "enforce", "provision", "terms", "policy", "patent",
    "copyright", "trademark", "lease", "property", "will", "estate", "trust",
    "corporate", "shareholder", "merger", "acquisition", "nda", "confidential",
    "indemnify", "warranty", "arbitration", "mediation", "judgment", "appeal",
    "summons", "subpoena", "affidavit", "deposition", "motion", "order",
    "what does", "what is", "explain", "define", "summarize", "who is",
    "when does", "how does", "what are", "list", "find", "tell me about",
]


# ─────────────────────────────────────────────
# PDF VALIDATION
# ─────────────────────────────────────────────
def validate_pdf_is_legal(text: str, filename: str = "") -> tuple[bool, str]:
    """
    Validates whether extracted PDF text belongs to a legal document.

    Returns:
        (True, "valid") if legal document detected.
        (False, reason_string) if not a legal document.
    """
    if not text or len(text.strip()) < 100:
        return False, (
            "The uploaded document appears to be empty or contains insufficient "
            "readable text. Please ensure the PDF is not scanned/image-only and "
            "contains selectable text."
        )

    text_lower = text.lower()
    word_count = len(text.split())

    # Count legal keyword matches
    matched_keywords = [kw for kw in LEGAL_DOCUMENT_KEYWORDS if kw in text_lower]
    match_score = len(matched_keywords)
    keyword_density = match_score / max(word_count / 100, 1)  # matches per 100 words

    # Decision logic
    if match_score >= 8 or keyword_density >= 1.5:
        return True, "valid"

    if match_score >= 4:
        # Borderline — check filename for hints
        fname_lower = filename.lower()
        legal_fname_hints = [
            "contract", "agreement", "nda", "legal", "law", "policy",
            "terms", "lease", "deed", "will", "trust", "court", "order",
            "brief", "motion", "statute", "regulation", "act",
        ]
        if any(hint in fname_lower for hint in legal_fname_hints):
            return True, "valid"

    # Not enough legal content
    if match_score == 0:
        return False, (
            "⚠️ **Non-Legal Document Detected**\n\n"
            "The uploaded file does not appear to be a legal document. "
            "No legal terminology, clauses, or statutory language was found in the content.\n\n"
            "**This system only processes legal documents** such as:\n"
            "- Contracts & Agreements\n"
            "- Court Orders & Judgments\n"
            "- Legislation & Regulations\n"
            "- Legal Briefs & Motions\n"
            "- Wills, Trusts & Estate Documents\n"
            "- Corporate & Compliance Documents\n"
            "- Intellectual Property Filings\n\n"
            "Please upload a valid legal document to proceed."
        )
    else:
        return False, (
            f"⚠️ **Insufficient Legal Content Detected**\n\n"
            f"The document contains some legal terminology ({match_score} legal terms found), "
            f"but does not meet the threshold to be classified as a legal document.\n\n"
            f"**Matched terms:** {', '.join(matched_keywords[:6])}{'...' if len(matched_keywords) > 6 else ''}\n\n"
            f"This system is designed exclusively for professional legal documents. "
            f"Please ensure you are uploading a complete legal document."
        )


# ─────────────────────────────────────────────
# CHAT QUERY VALIDATION
# ─────────────────────────────────────────────
def validate_chat_query(query: str) -> tuple[bool, str]:
    """
    Validates whether a user's chat message is legal in nature.

    Returns:
        (True, "valid") if the query is appropriate for a legal chatbot.
        (False, disclaimer_string) if off-topic or inappropriate.
    """
    if not query or len(query.strip()) < 3:
        return False, "Please enter a valid question."

    query_lower = query.lower().strip()

    # Check for harmful / illegal content patterns
    harmful_patterns = [
        r"\b(how to commit|how to avoid law|evade|launder|bribe|forge|falsify)\b",
        r"\b(kill|harm|attack|threaten|stalk|harass)\b",
        r"\b(make a bomb|weapon|explosive|drug synthesis)\b",
    ]
    for pattern in harmful_patterns:
        if re.search(pattern, query_lower):
            return False, (
                "⚠️ **Query Not Permitted**\n\n"
                "Your query appears to request information that involves illegal activities "
                "or content that violates ethical and legal standards.\n\n"
                "This legal assistant is designed to provide information strictly within "
                "lawful and professional boundaries. It cannot assist with queries that "
                "involve, facilitate, or promote illegal conduct.\n\n"
                "Please rephrase your question within a legitimate legal context."
            )

    # Check if query has legal intent
    has_legal_intent = any(kw in query_lower for kw in LEGAL_QUERY_KEYWORDS)

    # Check for non-legal patterns
    for pattern in NON_LEGAL_QUERY_PATTERNS:
        if re.search(pattern, query_lower):
            if not has_legal_intent:
                return False, (
                    "⚠️ **Out-of-Scope Query**\n\n"
                    "Your question does not appear to be related to legal matters or the "
                    "uploaded legal documents.\n\n"
                    "This assistant is a **specialized legal document intelligence system** "
                    "and can only respond to queries pertaining to:\n"
                    "- Content within the uploaded legal documents\n"
                    "- Legal terminology, clauses, and provisions\n"
                    "- Rights, obligations, and legal interpretations\n"
                    "- Compliance and regulatory questions\n\n"
                    "Please ask a question relevant to the uploaded legal documents."
                )

    # Short generic queries need at least some legal signal
    if len(query_lower.split()) <= 4 and not has_legal_intent:
        return False, (
            "⚠️ **Query Too Vague**\n\n"
            "Please provide a specific question related to the uploaded legal documents. "
            "For example: *'What are the termination clauses in this contract?'* or "
            "*'Who are the parties bound by this agreement?'*"
        )

    return True, "valid"