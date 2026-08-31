import re
from datetime import datetime


def normalize_name(name: str) -> str:
    """
    Normalize a person's name for comparison.

    Example:
        "Mohd. Irfan Sheikh" -> "MOHD IRFAN SHEIKH"
    """

    if not name:
        return ""

    name = name.upper()

    # Remove punctuation
    name = re.sub(r"[^A-Z0-9\s]", " ", name)

    # Remove extra spaces
    name = re.sub(r"\s+", " ", name).strip()

    return name


def normalize_address(address: str) -> str:
    """
    Normalize an address for fuzzy comparison.
    """

    if not address:
        return ""

    address = address.upper()

    # Replace common separators
    address = address.replace(",", " ")
    address = address.replace(".", " ")

    # Normalize common abbreviations
    replacements = {
        "ROAD": "RD",
        "STREET": "ST",
        "AVENUE": "AVE",
        "LANE": "LN",
        "APARTMENT": "APT",
    }

    words = address.split()

    normalized_words = [
        replacements.get(word, word)
        for word in words
    ]

    # Remove extra spaces
    return " ".join(normalized_words)


def normalize_dob(dob: str) -> str:
    """
    Normalize DOB into YYYY-MM-DD.

    Supports common formats such as:
        1995-03-12
        12-03-1995
        12/03/1995
    """

    if not dob:
        return ""

    dob = dob.strip()

    formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%d.%m.%Y",
    ]

    for fmt in formats:

        try:
            date = datetime.strptime(dob, fmt)

            return date.strftime("%Y-%m-%d")

        except ValueError:
            continue

    # Return original value if format is unknown
    return dob