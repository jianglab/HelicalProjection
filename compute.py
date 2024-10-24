import os, pathlib
import numpy as np
import pandas as pd
import helicon

@helicon.cache(
    cache_dir=str(helicon.cache_dir/"downloads"), expires_after=7, verbose=0
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


@helicon.cache(expires_after=7, cache_dir=helicon.cache_dir/"helicalProjection", verbose=0)
def symmetrize_project_one_map(data, apix, twist, rise, csym, map_name, image_query, image_query_label, image_query_apix, rescale_apix, length_xy_factor, match_sf):
    if abs(twist) < 1e-3:
        return None

    nz, ny, nx = data.shape
    if rescale_apix:
        new_apix = image_query_apix
        if abs(twist)<90:
            pitch = 360/abs(twist) * rise 
        else:
            pitch = 360/(180-abs(twist)) * rise
        image_ny, image_nx = image_query.shape
        length = int(pitch / new_apix + image_nx * length_xy_factor)//2*2
        new_size = (length, image_ny, image_ny)

        data_work = helicon.low_high_pass_filter(data, low_pass_fraction=apix/new_apix)
    else:
        new_apix = apix
        new_size = (nz, ny, nx)
        data_work = data

    profile_1d = np.sum(data_work, axis=(1, 2))
    threshold = 0.01 * np.max(profile_1d)
    non_zero_indices = np.where(profile_1d > threshold)[0]
    first_non_zero = non_zero_indices[0]
    last_non_zero = non_zero_indices[-1]
    max_fraction = min(last_non_zero - len(profile_1d) // 2, len(profile_1d) // 2 - first_non_zero) / len(profile_1d)
    assert max_fraction>0
    fraction = min(max_fraction, 5 * rise / (nz * apix))
    
    data_sym = helicon.apply_helical_symmetry(
        data = data_work,
        apix = apix,
        twist_degree = twist,
        rise_angstrom = rise,
        csym = csym,
        fraction = fraction,
        new_size = new_size,
        new_apix = new_apix,
        cpu = helicon.available_cpu()
    )
    proj = data_sym.sum(axis=2).T

    if match_sf:
        proj = helicon.match_structural_factors(data=proj, apix=new_apix, data_target=image_query, apix_target=new_apix)
        
    flip, rotation_angle, shift_cartesian, similarity_score, aligned_image_moving = helicon.align_images(image_moving=image_query, image_ref=proj, angle_range=15, check_polarity=True, check_flip=True, return_aligned_moving_image=True) 

    return (flip, rotation_angle, shift_cartesian, similarity_score, aligned_image_moving, image_query_label, proj, map_name)
