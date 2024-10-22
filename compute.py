import os, pathlib
import numpy as np
import pandas as pd
import helicon

from persist_cache import cache

@cache(
    name="emdb_projection", dir=str(helicon.cache_dir/"downloads"), expiry=7 * 24 * 60 * 60
)  # 7 days
def get_images_from_url(url):
    url_final = get_direct_url(url)  # convert cloud drive indirect url to direct url
    fileobj = download_file_from_url(url_final)
    if fileobj is None:
        raise ValueError(
            f"ERROR: {url} could not be downloaded. If this url points to a cloud drive file, make sure the link is a direct download link instead of a link for preview"
        )
    data = get_images_from_file(fileobj.name)
    return data


def get_images_from_file(imageFile):
    import mrcfile

    with mrcfile.open(imageFile) as mrc:
        apix = float(mrc.voxel_size.x)
        data = mrc.data
    return data, round(apix, 4)


def download_file_from_url(url):
    import tempfile
    import requests
    import os

    if pathlib.Path(url).is_file():
        return open(url, "rb")
    try:
        filesize = get_file_size(url)
        local_filename = url.split("/")[-1]
        suffix = "." + local_filename
        fileobj = tempfile.NamedTemporaryFile(suffix=suffix)
        with requests.get(url) as r:
            r.raise_for_status()  # Check for request success
            fileobj.write(r.content)
        return fileobj
    except requests.exceptions.RequestException as e:
        print(e)
        return None


def get_direct_url(url):
    import re

    if url.startswith("https://drive.google.com/file/d/"):
        hash = url.split("/")[5]
        return f"https://drive.google.com/uc?export=download&id={hash}"
    elif url.startswith("https://app.box.com/s/"):
        hash = url.split("/")[-1]
        return f"https://app.box.com/shared/static/{hash}"
    elif url.startswith("https://www.dropbox.com"):
        if url.find("dl=1") != -1:
            return url
        elif url.find("dl=0") != -1:
            return url.replace("dl=0", "dl=1")
        else:
            return url + "?dl=1"
    elif url.find("sharepoint.com") != -1 and url.find("guestaccess.aspx") != -1:
        return url.replace("guestaccess.aspx", "download.aspx")
    elif url.startswith("https://1drv.ms"):
        import base64

        data_bytes64 = base64.b64encode(bytes(url, "utf-8"))
        data_bytes64_String = (
            data_bytes64.decode("utf-8").replace("/", "_").replace("+", "-").rstrip("=")
        )
        return (
            f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
        )
    else:
        return url


def get_file_size(url):
    import requests

    response = requests.head(url)
    if "Content-Length" in response.headers:
        file_size = int(response.headers["Content-Length"])
        return file_size
    else:
        return None
    

def estimate_diameter(data):
    from scipy.optimize import curve_fit

    y = np.max(data, axis=1)
    n = len(y)
    x = np.arange(n) - n // 2
    lower_bounds = [0, -n // 2, 0, min(y)]
    upper_bounds = [max(y), n // 2, n // 2, max(y)]

    def gaussian(x, amp, center, sigma, background):
        return amp * np.exp(-(((x - center) / sigma) ** 2)) + background

    (amp, center, sigma, background), _ = curve_fit(
        gaussian, x, y, bounds=(lower_bounds, upper_bounds)
    )
    # diameter = (abs(center) + sigma*1.731) *2   # exp(-1.731**2) = 0.05
    diameter = sigma * 1.731 * 2  # exp(-1.731**2) = 0.05
    return diameter  # pixel    