# TODO(rcadene): obsolete remove
import os
import zipfile

import gdown


def download():
    url = "https://drive.google.com/uc?id=1nhxpykGtPDhmQKm-_B8zBSywVRdgeVya"
    download_path = "data.zip"
    gdown.download(url, download_path, quiet=False)
    print("Extracting...")
    with zipfile.ZipFile(download_path, "r") as zip_f:
        for member in zip_f.namelist():
            if member.startswith("data/xarm") and member.endswith(".pkl"):
                print(member)
                zip_f.extract(member=member)
    os.remove(download_path)


if __name__ == "__main__":
    download()
