import os, pathlib
import numpy as np
import pandas as pd
import helicon


class MapInfo:
    def __init__(self, data=None, filename=None, url=None, emd_id=None, label="", apix=None, twist=None, rise=None, csym=1):
        non_nones = [p for p in [data, filename, url, emd_id] if p is not None]
        if len(non_nones)>1:
            raise ValueError(f"MapInfo(): only one of these parameters can be set: data, filename, url, emd_id")
        elif len(non_nones)<1:
            raise ValueError(f"MapInfo(): one of these parameters must be set: data, filename, url, emd_id")
        self.data = data
        self.filename = filename
        self.url = url
        self.emd_id = emd_id
        self.label = label
        self.apix = apix
        self.twist = twist
        self.rise = rise
        self.csym = csym

    def __repr__(self):
        return (f"MapInfo(label={self.label}, emd_id={self.emd_id}, "
                f"twist={self.twist}, rise={self.rise}, csym={self.csym}, "
                f"apix={self.apix})")
        
    def get_data(self):
        if self.data is not None:
            return self.data, self.apix
        if isinstance(self.filename, str) and len(self.filename) and pathlib.Path(self.filename).exists():
            self.data, self.apix = get_images_from_file(self.filename)
            return self.data, self.apix
        if isinstance(self.url, str) and len(self.url):
            self.data, self.apix = get_images_from_url(self.url)
            return self.data, self.apix
        if isinstance(self.emd_id, str) and len(self.emd_id):
            emdb = helicon.dataset.EMDB()
            self.data, self.apix = emdb(self.emd_id)
            return self.data, self.apix
        raise ValueError(f"MapInfo.get_data(): failed to obtain data")


@helicon.cache(
    cache_dir=str(helicon.cache_dir / "helicalProjection"), expires_after=7, verbose=0
)  # 7 days
def get_images_from_url(url):
    url_final = get_direct_url(url)  # convert cloud drive indirect url to direct url
    fileobj = download_file_from_url(url_final)
    if fileobj is None:
        raise ValueError(
            f"ERROR: {url} could not be downloaded. If this url points to a cloud drive file, make sure the link is a direct download link instead of a link for preview"
        )
    data, apix = get_images_from_file(fileobj.name)
    return data, apix


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
    

def estimate_rotation(data, angle_range=30):
    from scipy.optimize import minimize_scalar

    data_work = helicon.threshold_data(data, thresh_fraction=0.2)

    def rotation_score(angle):
        rotated = helicon.transform_image(image=data_work, rotation=angle)
        rotated_yflip = rotated[:, ::-1]
        rotated_xflip = rotated[::-1, :]
        rotated_yxflip = rotated[::-1, ::-1]

        from itertools import combinations
        pairs = list(combinations([rotated, rotated_yflip, rotated_xflip, rotated_yxflip], 2))
        score = -np.sum([helicon.cross_correlation_coefficient(*p) for p in pairs])
        return score

    result = minimize_scalar(
        rotation_score, bounds=(-angle_range, angle_range), method="bounded"
    )
    return result.x

 
def estimate_diameter(data, return_center=False):
    from scipy.optimize import curve_fit

    y = np.max(data, axis=1)
    n = len(y)
    x = np.arange(n) - n // 2
    lower_bounds = [     0, -n // 2,      0, min(y)]
    upper_bounds = [max(y),  n // 2, n // 2, max(y)]

    def gaussian(x, amp, center, sigma, background):
        return amp * np.exp(-np.power((x - center) / sigma, 2)) + background

    (amp, center, sigma, background), _ = curve_fit(
        gaussian, x, y, bounds=(lower_bounds, upper_bounds)
    )
    #diameter = sigma * 1.731  # exp(-1.731**2) = 0.05
    diameter = sigma * 2.146  # exp(-2.146**2) = 0.01
    if return_center:
        return diameter, center  # pixel    
    else:
        return diameter  # pixel    



@helicon.cache(expires_after=7, cache_dir=helicon.cache_dir / "helicalProjection", verbose=0)
def get_one_map_xyz_projects(map_info, length_z, map_projection_xyz_choices):
    label = map_info.label
    try:
        data, apix = map_info.get_data()
    except Exception as e:
        if map_info.filename:
            msg = f"Failed to obtain uploaded map {label}"
        elif map_info.url:
            msg = f"Failed to download the map from {map_info.url}"
        elif map_info.emd_id:
            msg = f"Failed to download the map from EMDB for {map_info.emd_id}"
        raise ValueError(msg)
    
    images = []
    image_labels = []
    if 'z' in map_projection_xyz_choices:
        rise = map_info.rise
        if rise>0:
            images += [helicon.crop_center_z(data, n=max(1, int(0.5 + length_z * rise / apix))).sum(axis=0)]
        else:
            images += [data.sum(axis=0)]
        image_labels += [label + ':Z']
    if 'y' in map_projection_xyz_choices:
        images += [data.sum(axis=1)]
        image_labels += [label + ':Y']
    if 'x' in map_projection_xyz_choices:
        images += [data.sum(axis=2)]
        image_labels += [label + ':X']
        
    return images, image_labels


@helicon.cache(expires_after=7, cache_dir=helicon.cache_dir / "helicalProjection", verbose=0)
def symmetrize_project_align_one_map(map_info, image_query, image_query_label, image_query_apix, rescale_apix, length_xy_factor, match_sf, angle_range, scale_range):
    if abs(map_info.twist) < 1e-3:
        return None
    
    try:
        data, apix = map_info.get_data()
    except:
        return None

    twist = map_info.twist
    rise = map_info.rise
    csym = map_info.csym
    label = map_info.label
    
    nz, ny, nx = data.shape
    if rescale_apix:
        image_ny, image_nx = image_query.shape
        new_apix = image_query_apix
        twist_work = helicon.set_to_periodic_range(twist, min=-180, max=180)
        if abs(twist_work)<90:
            pitch = 360/abs(twist_work) * rise 
        elif abs(twist_work)<180:
            pitch = 360/(180-abs(twist_work)) * rise
        else:
            pitch = image_nx * new_apix
        length = int(pitch / new_apix + image_nx * length_xy_factor)//2*2
        new_size = (length, image_ny, image_ny)

        data_work = helicon.low_high_pass_filter(data, low_pass_fraction=apix/new_apix)
    else:
        new_apix = apix
        new_size = (nz, ny, nx)
        data_work = data

    fraction = 5 * rise / (nz * apix)
    
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
        
    flip, scale, rotation_angle, shift_cartesian, similarity_score, aligned_image_moving = helicon.align_images(image_moving=image_query, image_ref=proj, scale_range=scale_range, angle_range=angle_range, check_polarity=True, check_flip=True, return_aligned_moving_image=True) 

    return (flip, scale, rotation_angle, shift_cartesian, similarity_score, aligned_image_moving, image_query_label, proj, label)
