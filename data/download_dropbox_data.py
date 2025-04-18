import dropbox
import os
import re
from legalbenchrag.utils.credentials import credentials

# Dropbox shared link
SHARED_LINK = "https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4?rlkey=5n8zrbk4c08lbit3iiexofmwg&st=0hu354cq&dl=0"

# Local path to download into
LOCAL_DOWNLOAD_PATH = r"C:\dev\LLM4Law\legalbenchrag\data"

# Load token
DROPBOX_ACCESS_TOKEN = credentials.dropbox.token.get_secret_value()
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

# Replace invalid Windows filename characters
def sanitize_filename(filename):
    return re.sub(r'[\\/:*?"<>|]', '_', filename)

# Recursively download from shared folder
def download_shared_folder(shared_link, local_path, subpath=""):
    os.makedirs(local_path, exist_ok=True)

    try:
        result = dbx.files_list_folder(
            path=f"/{subpath}" if subpath else "",
            shared_link=dropbox.files.SharedLink(url=shared_link)
        )

        for entry in result.entries:
            sanitized_name = sanitize_filename(entry.name)

            if isinstance(entry, dropbox.files.FileMetadata):
                local_file_path = os.path.join(local_path, sanitized_name)
                print(f"Downloading: {entry.name} -> {local_file_path}")

                path_in_shared = f"/{subpath}/{entry.name}" if subpath else f"/{entry.name}"

                _, res = dbx.sharing_get_shared_link_file(
                    url=shared_link,
                    path=path_in_shared
                )
                with open(local_file_path, "wb") as f:
                    f.write(res.content)

            elif isinstance(entry, dropbox.files.FolderMetadata):
                print(f"Entering folder: {entry.name}")
                new_local_path = os.path.join(local_path, sanitized_name)
                new_subpath = f"{subpath}/{entry.name}" if subpath else entry.name
                download_shared_folder(shared_link, new_local_path, new_subpath)

    except Exception as e:
        print(f"Error: {e}")


# Start download
download_shared_folder(SHARED_LINK, LOCAL_DOWNLOAD_PATH)
