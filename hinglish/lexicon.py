"""
lexicon.py — Word lists for Hinglish (Romanised Hindi mixed with English).

Pure data plus two small helpers; no imports from the rest of the project, so every
module (including llm_client) can import it without a circular import.

Keep the lists versioned like the Hindi glossary: a change here changes how questions
are detected and normalised.
"""

import re

# ---------------------------------------------------------------------------
# Spelling variants → one standard Roman spelling
#
# Romanised Hindi has no fixed spelling ("hu", "hun", "hoon"). Collapsing the variants
# gives every spelling of the same question the SAME cache key, so it is converted to
# Hindi once and always reaches the same canonical English question.
# ---------------------------------------------------------------------------

SPELLING_VARIANTS = {
    # pronouns
    "mai": "main", "mei": "main", "mjhe": "mujhe", "muje": "mujhe", "mujhey": "mujhe",
    "mujhko": "mujhe", "mera": "mera", "meraa": "mera", "apko": "aapko", "aap": "aap", "ap": "aap",
    # auxiliaries
    "hu": "hoon", "hun": "hoon", "hoo": "hoon", "hoon": "hoon", "hn": "hain", "hein": "hain",
    "hy": "hai", "haii": "hai",
    # question words
    "kia": "kya", "kyaa": "kya", "kyu": "kyun", "kyon": "kyun", "kyoon": "kyun", "kiyu": "kyun",
    "kese": "kaise", "kaisey": "kaise", "kitana": "kitna", "konsa": "kaunsa", "kab": "kab",
    # negation
    "nahin": "nahi", "nai": "nahi", "nhi": "nahi", "nahee": "nahi", "nahi": "nahi",
    # modals
    "skta": "sakta", "skti": "sakti", "skte": "sakte", "sakta": "sakta",
    "chahie": "chahiye", "chaiye": "chahiye", "chahiyee": "chahiye", "chaahiye": "chahiye",
    "chaahie": "chahiye", "chahiya": "chahiye",
    # verbs
    "khaana": "khana", "khaa": "kha", "kr": "kar", "krna": "karna", "krne": "karne", "karu": "karun",
    "kru": "karun", "lu": "loon", "lun": "loon",
    # time / quantity
    "pahle": "pehle", "phle": "pehle", "pehele": "pehle", "pehley": "pehle", "pahele": "pehle",
    "bohot": "bahut", "bahot": "bahut", "bht": "bahut", "jyada": "zyada",
    "jada": "zyada", "zyaada": "zyada", "jyaada": "zyada", "thik": "theek",
    # connectives
    "agr": "agar", "lkin": "lekin", "bhee": "bhi", "kyunki": "kyunki", "kyuki": "kyunki",
    "kyonki": "kyunki", "isliye": "isliye", "isliy": "isliye",
    # food / body
    "bhookh": "bhook", "bhuk": "bhook", "bhukh": "bhook", "wajan": "vajan", "vazan": "vajan",
    "cheeni": "cheeni", "chini": "cheeni", "gur": "gud",
}

# ---------------------------------------------------------------------------
# Marker words used by the detector (standard spellings, after SPELLING_VARIANTS)
#
# STRONG markers are not English words. WEAK markers are also English words ("main",
# "to", "hi"), so they count half and can never make a sentence Hinglish on their own.
# ---------------------------------------------------------------------------

STRONG_MARKERS = {
    # question words
    "kya", "kyun", "kaise", "kab", "kitna", "kitni", "kitne", "kaun", "kaunsa", "kahan",
    # auxiliaries and copulas
    "hai", "hain", "hoon", "tha", "thi", "hota", "hoti", "hote", "hua", "hui", "hoga", "hogi",
    # pronouns
    "mujhe", "mera", "meri", "mere", "humein", "hamein", "aapko", "aapka", "aapki",
    "tum", "tumhe", "unhe", "unko", "isko", "usko", "yeh", "woh", "vo",
    # negation and modals
    "nahi", "sakta", "sakti", "sakte", "chahiye", "padega", "padta",
    # verbs
    "karna", "karne", "karo", "karun", "karein", "kare", "kar", "kiya", "karta", "karti",
    "kha", "khana", "khaye", "khau", "khaun", "khata", "khati", "khate", "peena", "piyu",
    "lena", "loon", "lein", "liya", "leta", "leti", "dena", "diya", "jana", "jata", "jati",
    "lagta", "lagti", "lagte", "lagi", "raha", "rahi", "rahe", "gaya", "gayi", "chalna",
    "ghatana", "badhana", "rakhna", "sochna",
    # postpositions and connectives
    "ke", "ka", "ko", "se", "mein", "wala", "wali", "wale", "agar", "toh", "lekin", "aur",
    "bhi", "sirf", "kyunki", "isliye", "saath", "bina", "wajah", "liye",
    # time and quantity
    "pehle", "baad", "abhi", "kabhi", "roz", "subah", "shaam", "raat", "bahut", "zyada",
    "thoda", "theek", "kam",
    # food and body
    "bhook", "vajan", "roti", "chawal", "sabzi", "dahi", "doodh", "cheeni", "shakkar", "gud",
    "mithai", "neend",
}

WEAK_MARKERS = {"main", "me", "to", "hi", "ki", "ye", "so", "par", "ya", "na", "ho", "din", "pet", "le",
                "mat", "hum"}

# ---------------------------------------------------------------------------
# Protected medical terms
#
# Kept verbatim (in Latin script, with this spelling) through every Hinglish step:
# masked before the Roman→Devanagari conversion and required in every Hinglish answer
# whose English source uses them. A mistranslated "remission" or "HbA1c" changes advice.
# Longest first, so "type 2 diabetes" wins over "diabetes".
# ---------------------------------------------------------------------------

PROTECTED_TERMS = [
    "type 2 diabetes", "type 1 diabetes", "insulin resistance", "waist circumference",
    "Dixit Lifestyle", "prediabetes", "diabetes", "diabetic", "HbA1c", "insulin", "obesity",
    "remission", "BMI", "PCOS", "glimepiride", "sulfonylurea", "metformin", "hypoglycaemia",
    "hypoglycemia",
]
PROTECTED_TERMS.sort(key=len, reverse=True)

_TERM_PATTERNS = [(term, re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", re.IGNORECASE))
                  for term in PROTECTED_TERMS]


def term_patterns() -> list[tuple[str, re.Pattern]]:
    """(standard spelling, compiled case-insensitive whole-word pattern), longest term first."""
    return _TERM_PATTERNS


def find_protected_terms(text: str) -> set[str]:
    """Standard spellings of the protected terms that appear in `text` (whole words, any case)."""
    return {term for term, pattern in _TERM_PATTERNS if pattern.search(text or "")}
