import re


# Define characters typically illegal in Windows filenames and replacement
ILLEGAL_FILENAME_CHARS = r'[\\/:*?"<>|]'
REPLACEMENT_CHAR = '_'


def sanitize_filename(filename: str) -> str:
    """Replaces characters illegal in Windows filenames with underscores."""
    return re.sub(ILLEGAL_FILENAME_CHARS, REPLACEMENT_CHAR, filename)
