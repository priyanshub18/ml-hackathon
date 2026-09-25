"""
Vectorized text normalization for business names and addresses.

All public functions work on entire pandas Series (not row-by-row)
so they run in seconds on millions of records.
"""

import re
import unicodedata
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Legal suffix maps — sorted longest-first for greedy matching
# ---------------------------------------------------------------------------
_LEGAL_SUFFIX_MAP = {
    # English multi-word (must come before single-word variants)
    "private limited": "pvt ltd",
    "pvt limited": "pvt ltd",
    "pvt. limited": "pvt ltd",
    "pvt. ltd.": "pvt ltd",
    "pvt. ltd": "pvt ltd",
    "pvt ltd": "pvt ltd",
    "p. ltd.": "pvt ltd",
    "p ltd": "pvt ltd",
    "limited liability partnership": "llp",
    "limited liability company": "llc",
    "incorporated": "inc",
    "incorporation": "inc",
    "corporation": "corp",
    "limited": "ltd",
    "company": "co",
    # French (accent-stripped versions, so ç → c, é → e etc.)
    "societe a responsabilite limitee": "sarl",
    "societe par actions simplifiee unipersonnelle": "sasu",
    "societe par actions simplifiee": "sas",
    "societe anonyme": "sa",
    "entreprise unipersonnelle a responsabilite limitee": "eurl",
    "societe en nom collectif": "snc",
    "groupement d interet economique": "gie",
    "societe civile immobiliere": "sci",
    "societe civile": "sc",
    # Abbreviation forms (already canonical)
    "sarl": "sarl",
    "sasu": "sasu",
    "sas": "sas",
    "eurl": "eurl",
    "snc": "snc",
    "gie": "gie",
    "sci": "sci",
    "llp": "llp",
    "llc": "llc",
    "inc": "inc",
    "corp": "corp",
    "ltd": "ltd",
}

# Build patterns sorted longest → shortest
_SUFFIX_KEYS = sorted(_LEGAL_SUFFIX_MAP.keys(), key=len, reverse=True)

# Single combined regex: captures the first matching suffix
# \b word boundary on both sides so "ltd" in "ltd company" matches correctly
_SUFFIX_RE_STR = r"(?:^|(?<=\s))(" + "|".join(re.escape(k) for k in _SUFFIX_KEYS) + r")(?:\s|$)"
_SUFFIX_RE = re.compile(_SUFFIX_RE_STR)

# Reversed map: canonical suffix → set of raw forms (not needed but kept for reference)
# Replacement map (what to substitute the match with)
_SUFFIX_REPLACE = {k: _LEGAL_SUFFIX_MAP[k] for k in _SUFFIX_KEYS}

# ---------------------------------------------------------------------------
# Address abbreviation map (applied with word-boundary regex)
# ---------------------------------------------------------------------------
_ADDR_ABBREVS = {
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\brd\b": "road",
    r"\bblvd\b": "boulevard",
    r"\bbd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bsq\b": "square",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bfwy\b": "freeway",
    r"\bexpy\b": "expressway",
}
# Combined single-pass regex for address abbreviations
_ADDR_RE = re.compile("|".join(f"(?P<g{i}>{pat})" for i, pat in enumerate(_ADDR_ABBREVS)))
_ADDR_FULL = list(_ADDR_ABBREVS.values())
_ADDR_PATS = list(_ADDR_ABBREVS.keys())


# ---------------------------------------------------------------------------
# Vectorized helpers
# ---------------------------------------------------------------------------

def _strip_accents_series(s: pd.Series) -> pd.Series:
    """NFKD normalization + remove diacritics, vectorized via unicodedata."""
    def _strip(text):
        if not isinstance(text, str):
            return ""
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    return s.apply(_strip)


def normalize_name_series(names: pd.Series):
    """
    Vectorized normalize_name for an entire Series.
    Returns tuple of three Series: (core_name, suffix, full_norm).
    """
    # Convert nan → ""
    text = names.fillna("").astype(str)

    # Strip accents
    text = _strip_accents_series(text)
    # Lowercase
    text = text.str.lower()
    # & → and
    text = text.str.replace(r"&", " and ", regex=False)
    # Remove punctuation except spaces/hyphens-between-words
    text = text.str.replace(r"[^\w\s]", " ", regex=True)
    # Collapse whitespace
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()

    # Suffix extraction (vectorized via apply on each element — unavoidable
    # because suffix removal modifies the text differently per row)
    def _extract_suffix(t):
        m = _SUFFIX_RE.search(t)
        if m:
            matched = m.group(1)
            canonical = _LEGAL_SUFFIX_MAP.get(matched, matched)
            # Remove the matched suffix from the text
            core = (t[: m.start()].strip() + " " + t[m.end() :].strip()).strip()
            core = re.sub(r"\s+", " ", core).strip()
            return core, canonical
        return t, ""

    results = text.apply(_extract_suffix)
    core   = results.apply(lambda x: x[0])
    suffix = results.apply(lambda x: x[1])
    full   = core + suffix.apply(lambda s: (" " + s) if s else "")
    full   = full.str.strip()

    return core, suffix, full


def normalize_address_series(addresses: pd.Series):
    """
    Vectorized normalize_address.
    Returns tuple of two Series: (norm_addr, numeric_tokens list).
    """
    addr = addresses.fillna("").astype(str)
    # Replace nan / null strings
    addr = addr.str.replace(r"^(nan|null|none)$", "", regex=True)

    # Strip accents (vectorized)
    addr = _strip_accents_series(addr)

    # Lowercase
    addr = addr.str.lower()

    # Expand abbreviations (compiled regex, single pass per row via str.replace loops)
    # This is O(n_abbrevs) passes but each pass is vectorized
    for pat, full in _ADDR_ABBREVS.items():
        addr = addr.str.replace(pat, full, regex=True)

    # Remove punctuation
    addr = addr.str.replace(r"[,./\-#()\"']", " ", regex=True)
    addr = addr.str.replace(r"\s+", " ", regex=True).str.strip()

    # Numeric token extraction — vectorized via findall
    def _extract_nums(text):
        if not text:
            return []
        nums = re.findall(r"\b\d+\b", text)
        return list({n for n in nums if len(n) >= 2})   # skip single digits

    numeric_tokens = addr.apply(_extract_nums)

    return addr, numeric_tokens


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized columns to a dataframe containing 'business_name' and
    'business_address'. Adds:
        norm_core, norm_suffix, norm_name  — name columns
        norm_addr, numeric_tokens          — address columns
        is_non_latin                       — flag for non-ASCII name
    """
    df = df.copy()

    core, suffix, full = normalize_name_series(df["business_name"])
    df["norm_core"]   = core
    df["norm_suffix"] = suffix
    df["norm_name"]   = full

    norm_addr, num_toks = normalize_address_series(df["business_address"])
    df["norm_addr"]       = norm_addr
    df["numeric_tokens"]  = num_toks

    # Non-Latin flag (S1 is always Latin; S2/S3 may have Hindi/Tamil/etc.)
    df["is_non_latin"] = df["business_name"].apply(
        lambda x: bool(re.search(r"[^\x00-\x7F]", str(x)))
    )

    return df
