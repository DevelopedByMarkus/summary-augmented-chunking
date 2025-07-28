import re
import os


# Define characters typically illegal in Windows filenames and replacement
ILLEGAL_FILENAME_CHARS = r'[\\/:*?"<>|]'
REPLACEMENT_CHAR = '_'


def sanitize_filename(filename: str, ignore_dirs: str = None) -> str:
    """
    Replaces characters illegal in Windows filenames with underscores.

    If ignore_dirs is provided, it will not sanitize that leading part
    of the filename string.
    """
    if ignore_dirs and filename.startswith(ignore_dirs):
        # Separate the directory part from the filename part
        base_path = ignore_dirs
        file_part = filename[len(ignore_dirs):]

        # Sanitize only the filename part
        sanitized_file_part = re.sub(ILLEGAL_FILENAME_CHARS, REPLACEMENT_CHAR, file_part)

        # Return the original directory path with the sanitized filename
        return os.path.join(base_path, sanitized_file_part)
    else:
        if ignore_dirs:
            print(f"WARNING: ignore_dirs param given, but '{filename}' does not start with '{ignore_dirs}'.")
        # If no directory is ignored, sanitize the whole string
        return re.sub(ILLEGAL_FILENAME_CHARS, REPLACEMENT_CHAR, filename)
