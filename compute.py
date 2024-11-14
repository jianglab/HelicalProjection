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
    url_final = helicon.get_direct_url(url)  # convert cloud drive indirect url to direct url
    fileobj = helicon.download_file_from_url(url_final)
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

def get_amyloid_n_sub_1_symmetry(twist, rise, max_n=10):
    ret = 1
    for n in range(max_n, 1, -1):
        if not (4.5 < rise * n < 5):
            continue
        if abs(360 - abs(twist * n)) > 90:
            continue
        ret = n
        break
    return ret 

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
            rise *=  get_amyloid_n_sub_1_symmetry(twist=map_info.twist, rise=map_info.rise)
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
        
    flip, scale, rotation_angle, shift_cartesian, similarity_score, aligned_image_moving = helicon.align_images(image_moving=image_query, image_ref=proj, scale_range=scale_range, angle_range=angle_range, check_polarity=True, check_flip=True, return_aligned_moving_image=True) 

    if match_sf:
        proj = helicon.match_structural_factors(data=proj, apix=new_apix, data_target=aligned_image_moving, apix_target=new_apix)

    return (flip, scale, rotation_angle, shift_cartesian, similarity_score, aligned_image_moving, image_query_label, proj, label)
