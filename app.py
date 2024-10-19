from pathlib import Path
import numpy as np
import pandas as pd

import shiny
from shiny import reactive, req
from shiny.express import input, ui, render

import helicon

from . import compute

images_all = reactive.value([])
image_size = reactive.value(0)
image_apix = reactive.value(0)

displayed_image_ids = reactive.value([])
displayed_images = reactive.value([])
displayed_image_title = reactive.value("Select an image:")
displayed_image_labels = reactive.value([])

initial_selected_image_indices = reactive.value([0])
selected_images = reactive.value([])
selected_image_title = reactive.value("Selected image:")
selected_image_labels = reactive.value([])

emdb_df = reactive.value([])

maps = reactive.value([])
map_xyz_projections = reactive.value([])
map_xyz_projection_title = reactive.value("Map XYZ projections:")
map_xyz_projection_labels = reactive.value([])
map_xyz_projection_display_size = reactive.value(128)

map_side_projections = reactive.value([])
map_side_projection_title = reactive.value("Map side projections:")
map_side_projection_labels = reactive.value([])
map_side_projection_display_size = reactive.value(256)
 
ui.head_content(ui.tags.title("HelicalProjection"))
helicon.shiny.google_analytics(id="G-ELN1JJVYYZ")
helicon.shiny.setup_ajdustable_sidebar()
ui.tags.style(
    """
    * { font-size: 10pt; padding:0; border: 0; margin: 0; }
    aside {--_padding-icon: 10px;}
    """
)
urls = {
    "empiar-10940_job010": (
        "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/Class2D/job010/run_it020_classes.mrcs",
        "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-14046/map/emd_14046.map.gz"
    )
}
url_key = "empiar-10940_job010"

with ui.sidebar(
    width="33vw", style="display: flex; flex-direction: column; height: 100%;"
):
    with ui.navset_pill(id="tab"):  
        with ui.nav_panel("Input 2D Images"):
            with ui.div(id="input_image_files", style="display: flex; flex-direction: column; align-items: flex-start;"):
                ui.input_radio_buttons(
                    "input_mode_images",
                    "How to obtain the input images:",
                    choices=["upload", "url"],
                    selected="url",
                    inline=True,
                )
                
                @render.ui
                @reactive.event(input.input_mode_images)
                def create_input_image_files_ui():
                    displayed_images.set([])
                    ret = []
                    if input.input_mode_images() == 'upload':
                        ret.append(
                            ui.input_file(
                                "upload_images",
                                "Upload the input images in MRC format (.mrcs, .mrc)",
                                accept=[".mrcs", ".mrc"],
                                placeholder="mrcs or mrc file",
                            )                            
                        )
                    elif input.input_mode_images() == 'url':
                        ret.append(
                            ui.input_text(
                                "url_images",
                                "Download URL for a RELION or cryoSPARC image output mrc(s) file",
                                value=urls[url_key][0],
                            )
                        )
                    return ret

            with ui.div(id="image-selection", style="max-height: 80vh; overflow-y: auto;"):
                helicon.shiny.image_select(
                    id="select_images",
                    label=displayed_image_title,
                    images=displayed_images,
                    image_labels=displayed_image_labels,
                    image_size=reactive.value(128),
                    initial_selected_indices=initial_selected_image_indices,
                    allow_multiple_selection=False
                )

                @reactive.effect
                @reactive.event(input.select_images)
                def update_selected_images():
                    selected_images.set(
                        [displayed_images()[i] for i in input.select_images()]
                    )
                    selected_image_labels.set(
                        [displayed_image_labels()[i] for i in input.select_images()]
                    )

        with ui.nav_panel("Input 3D Maps"):
            with ui.div(id="input_map_files", style="display: flex; flex-direction: column; align-items: flex-start;"):
                ui.input_radio_buttons(
                    "input_mode_maps",
                    "How to obtain the 3D maps:",
                    choices=["upload", "url", "amyloid_atlas", "EMDB-helical", "EMDB"],
                    selected="url",
                    inline=True,
                )

                @render.ui
                @reactive.event(input.input_mode_maps)
                def create_input_map_files_ui():
                    maps.set([])
                    map_xyz_projections.set([])
                    map_side_projections.set([])
                    ret = []
                    if input.input_mode_maps() in ['upload', 'url']:
                        if input.input_mode_maps() == 'upload':
                            ret.append(
                                ui.input_file(
                                    "upload_map",
                                    "Upload the 3D map in MRC format (.mrc, .mrc.gz, .map, .map.gz)",
                                    accept=[".mrc", ".mrc.gz", ".map", ".map.gz"],
                                    placeholder="mrc file",
                                )
                            )

                        elif input.input_mode_maps() == 'url':
                            ret.append(
                                ui.input_text(
                                    "url_map",
                                    "Download URL for a map file in MRC format",
                                    value=urls[url_key][1],
                                )
                            )
                            
                        ret.append(
                            shiny.ui.layout_columns(
                                ui.input_numeric("twist", "Twist (°)", value=179.402, min=-180, max=180, step=1),
                                ui.input_numeric("rise", "Rise (Å)", value=2.378, min=0, step=1),
                                ui.input_numeric("csym", "Csym", value=1, min=1, step=1),
                                col_widths=[4,4,4], style="align-items: flex-end;"
                            )
                        )
                        
                    elif input.input_mode_maps() in ['amyloid_atlas', 'EMDB-helical', 'EMDB']:
                        emdb = helicon.dataset.EMDB()
                        cols = ["emdb_id", "pdb", "resolution", "twist", "rise", "csym", "title"]
                        if input.input_mode_maps() == 'amyloid_atlas':
                            emd_ids = emdb.amyloid_atlas_ids()
                        elif input.input_mode_maps() == 'EMDB-helical':
                            emd_ids = emdb.helical_structure_ids()
                        elif input.input_mode_maps() == 'EMDB':
                            emd_ids = emdb.emd_ids
                            cols = ["emdb_id", "pdb", "resolution", "title"]
                        df = emdb.meta.loc[emdb.meta["emd_id"].isin(emd_ids)]
                        df = df[cols].round(3)
                        emdb_df.set(df)

                    return ret
                
                with ui.panel_conditional("input.input_mode_maps === 'amyloid_atlas' || input.input_mode_maps === 'EMDB-helical' || input.input_mode_maps === 'EMDB'"):
                    @render.data_frame
                    @reactive.event(emdb_df, input.show_pdb)
                    def display_emdb_dataframe():
                        df = emdb_df()
                        if not input.show_pdb():
                            cols = [col for col in df.columns if col != "pdb"]
                            df = df[cols]
                        return render.DataGrid(
                            df,
                            selection_mode="rows",
                            filters=True,
                            height="40vh",
                            width="100%"
                        )
                
            with ui.div(style="max-height: 80vh; overflow-y: auto;"):
                helicon.shiny.image_select(
                    id="display_map_xyz_projections",
                    label=map_xyz_projection_title,
                    images=map_xyz_projections,
                    image_labels=map_xyz_projection_labels,
                    image_size=map_xyz_projection_display_size,
                    enable_selection=False
                )

        with ui.nav_panel("Parameters"):
            with ui.layout_columns(
                col_widths=[6], style="align-items: flex-end;"
            ):
                ui.input_checkbox_group(
                    "map_projection_xyz_choices",
                    "Show projections along:",
                    choices=['x', 'y', 'z'],
                    selected=['x', 'y', 'z'],
                    inline=True
                )

            with ui.layout_columns(
                col_widths=[6, 6, 6, 6], style="align-items: flex-end;"
            ):
                ui.input_checkbox(
                    "ignore_blank", "Ignore blank input images", value=True
                )

                ui.input_checkbox(
                    "show_pdb", "Show PDB ids in EMDB table", value=False
                )

                ui.input_checkbox(
                    "rescale_apix", "Resample to image pixel size", value=True
                )

                ui.input_checkbox(
                    "match_sf", "Apply matched-filter", value=False
                )

            with ui.layout_columns(
                col_widths=6, style="align-items: flex-end;"
            ):                    
                ui.input_numeric(
                    "map_xyz_projection_display_size",
                    "Map XYZ projection image size (pixel)",
                    min=32,
                    max=512,
                    value=128,
                    step=16,
                )
                ui.input_numeric(
                    "length_z",
                    "Z-projection length (x rise)",
                    min=0,
                    value=1,
                    step=1,
                )
                ui.input_numeric(
                    "length_xy",
                    "Side projection length (x pitch)",
                    min=0,
                    value=1.2,
                    step=0.1,
                )

title = "HelicalProjection: compare 2D images with helical structure projections"
ui.h1(title, style="font-weight: bold;")

with ui.div(style="display: flex; flex-direction: row; align-items: flex-start; gap: 10px;"):
    helicon.shiny.image_select(
        id="display_selected_image",
        label=selected_image_title,
        images=selected_images,
        image_labels=selected_image_labels,
        image_size=map_side_projection_display_size,
        justification="left",
        enable_selection=False,
    )

    with ui.div():
        ui.input_slider(
            "map_side_projection_display_size",
            "Map side projection image size (pixel)",
            min=32,
            max=1024,
            value=256,
            step=32,
        )
        
        @render.ui
        @reactive.event(maps)
        def display_action_button():
            req(len(maps()))
            return ui.input_task_button("generate_projections", label="Generate Projections", width="256px")

with ui.div(style="max-height: 80vh; overflow-y: auto;"):
    helicon.shiny.image_select(
        id="display_map_side_projections",
        label=map_side_projection_title,
        images=map_side_projections,
        image_labels=map_side_projection_labels,
        image_size=map_side_projection_display_size,
        justification="left",
        enable_selection=False
    )

ui.HTML(
    "<i><p>Developed by the <a href='https://jiang.bio.purdue.edu/HelicalProjection' target='_blank'>Jiang Lab</a>. Report issues to <a href='https://github.com/jianglab/HelicalProjection/issues' target='_blank'>HelicalProjection@GitHub</a>.</p></i>"
)

@reactive.effect
@reactive.event(input.input_mode_images, input.upload_images)
def get_image_from_upload():
    req(input.input_mode_images() == "upload")
    fileinfo = input.upload_images()
    req(fileinfo)
    image_file = fileinfo[0]["datapath"]
    try:
        data, apix = compute.get_images_from_file(image_file)
    except Exception as e:
        print(e)
        data, apix = None, 0
        m = ui.modal(
            f"failed to read the uploaded 2D images from {fileinfo[0]['name']}",
            title="File upload error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)
    images_all.set(data)
    image_size.set(min(data.shape))
    image_apix.set(apix)


@reactive.effect
@reactive.event(input.input_mode_images, input.url_images)
def get_images_from_url():
    req(input.input_mode_images() == "url")
    req(len(input.url_images()) > 0)
    url = input.url_images()
    try:
        data, apix = compute.get_images_from_url(url)
    except Exception as e:
        print(e)
        data, apix = None, 0
        m = ui.modal(
            f"failed to download 2D images from {input.url_images()}",
            title="File download error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)
    images_all.set(data)
    image_size.set(min(data.shape))
    image_apix.set(apix)


@reactive.effect
@reactive.event(images_all, input.ignore_blank)
def get_displayed_images():
    req(len(images_all()))
    data = images_all()
    n = len(data)
    ny, nx = data[0].shape[:2]
    images = [data[i] for i in range(n)]
    image_size.set(max(images[0].shape))

    display_seq_all = np.arange(n, dtype=int)
    if input.ignore_blank():
        included = []
        for i in range(n):
            image = images[display_seq_all[i]]
            if np.max(image) > np.min(image):
                included.append(display_seq_all[i])
        images = [images[i] for i in included]
    else:
        included = display_seq_all
    image_labels = [f"{i+1}" for i in included]

    displayed_image_ids.set(included)
    displayed_images.set(images)
    displayed_image_title.set(f"{len(images)}/{n} images | {nx}x{ny} pixels | {image_apix()} Å/pixel")
    displayed_image_labels.set(image_labels)


@reactive.effect
@reactive.event(input.input_mode_maps, input.upload_map, input.twist, input.rise, input.csym)
def get_map_from_upload():
    req(input.input_mode_maps() == "upload")
    fileinfo = input.upload_map()
    req(fileinfo)
    map_file = fileinfo[0]["datapath"]
    try:
        data, apix = compute.get_images_from_file(map_file)
    except Exception as e:
        print(e)
        data, apix = None, 0
        m = ui.modal(
            f"failed to read the uploaded 3D map from {fileinfo[0]['name']}",
            title="File upload error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)
    maps.set([(data, apix, input.twist(), input.rise(), input.csym(), fileinfo[0]["name"])])


@reactive.effect
@reactive.event(input.input_mode_maps, input.url_map, input.twist, input.rise, input.csym)
def get_map_from_url():
    req(input.input_mode_maps() == "url")
    req(len(input.url_map()) > 0)
    url = input.url_map()
    try:
        data, apix = compute.get_images_from_url(url)
    except Exception as e:
        print(e)
        data, apix = None, 0
        nx = 0
        m = ui.modal(
            f"failed to download 3D map from {input.url_map()}",
            title="File download error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)
    maps.set([(data, apix, input.twist(), input.rise(), input.csym(), url.split("/")[-1])])


@reactive.effect
def get_map_from_emdb():
    emdb_df_selected = display_emdb_dataframe.data_view(selected=True)
    req(len(emdb_df_selected) > 0)
    emdb = helicon.dataset.EMDB()
    maps_tmp = []
    for _, row in emdb_df_selected.iterrows():
        emdb_id = row['emdb_id']
        twist = row['twist'] if 'twist' in row and not pd.isna(row['twist']) else 0
        rise = row['rise'] if 'rise' in row and not pd.isna(row['rise']) else 0
        csym = int(row['csym'][1:]) if 'csym' in row and not pd.isna(row['csym']) else 1
        try:
            data, apix = emdb(emdb_id)
            maps_tmp.append((data, apix, twist, rise, csym, f"{emdb_id}"))
        except Exception as e:
            print(f"Failed to download map for {emdb_id}: {e}")
            m = ui.modal(
                f"Failed to download 3D map for {emdb_id}",
                title="EMDB Download Error",
                easy_close=True,
                footer=None,
            )
            ui.modal_show(m)
    map_side_projections.set([])
    maps.set(maps_tmp)
    
 
@reactive.effect
@reactive.event(maps, input.length_z, input.map_projection_xyz_choices)
def get_map_xyz_projections():
    req(maps() is not None)
    images = []
    image_labels = []

    for mi, m in enumerate(maps()):
        data, apix, twist, rise, csym, filename = m
        filename = filename.split('.')[0]
        
        if 'z' in input.map_projection_xyz_choices():
            if rise>0:
                images += [helicon.crop_center_z(data, n=max(1, int(0.5 + input.length_z() * rise / apix))).sum(axis=0)]
            else:
                images += [data.sum(axis=0)]
            image_labels += [filename + ':Z']
        if 'y' in input.map_projection_xyz_choices():
            images += [data.sum(axis=1)]
            image_labels += [filename + ':Y']
        if 'x' in input.map_projection_xyz_choices():
            images += [data.sum(axis=2)]
            image_labels += [filename + ':X']

        map_xyz_projection_labels.set(image_labels)
        map_xyz_projections.set(images)
    
@reactive.effect
@reactive.event(input.generate_projections, input.match_sf, input.rescale_apix, input.length_xy)
def get_map_side_projections():
    req(maps() is not None)
    images = []
    image_labels = []

    for mi, m in enumerate(maps()):
        data, apix, twist, rise, csym, filename = m
        if abs(twist) < 1e-3:
            m = ui.modal(
                f"WARNING: {twist=}°. Please set twist to a correct value for this structure",
                title="Twist value error",
                easy_close=True,
                footer=None,
            )
            ui.modal_show(m)
            return

        filename = filename.split('.')[0]

        nz, ny, nx = data.shape
        if input.rescale_apix():
            new_apix = image_apix()
            if abs(twist)<90:
                pitch = 360/abs(twist) * rise 
            else:
                pitch = 360/(180-abs(twist)) * rise 
            length = (int(pitch * input.length_xy() / new_apix)+2)//2*2
            new_size = (length, image_size(), image_size())

            data_work = helicon.low_high_pass_filter(data, low_pass_fraction=apix/new_apix)
        else:
            new_apix = apix
            new_size = (nz, ny, nx)
            data_work = data

        def projection(apix, twist, rise, csym, new_size, new_apix):
            data_sym = helicon.apply_helical_symmetry(
                data = data_work,
                apix = apix,
                twist_degree = twist,
                rise_angstrom = rise,
                csym = csym,
                fraction = min(0.333, 5 * rise/(nz*apix)),
                new_size = new_size,
                new_apix = new_apix,
                cpu = 1
            )
            proj = data_sym.sum(axis=2).T

            if input.match_sf():
                proj = helicon.match_structural_factors(proj, new_apix, data_target=selected_images()[0], apix_target=image_apix())
            return proj
        
        from persist_cache import cache
        @cache(
            name="emdb_projection", dir=str(helicon.cache_dir/"emdb"), expiry=100 * 24 * 60 * 60
        )  # 100 days
        def cached_projection(apix, twist, rise, csym, filename, new_size, new_apix):
            return projection(apix, twist, rise, csym, new_size, new_apix)
        
        if filename.lower().startswith("emd"):
            proj = cached_projection(apix, twist, rise, csym, filename, new_size, new_apix)
        else:
            proj = projection(apix, twist, rise, csym, new_size, new_apix)

        images += [proj]
        image_labels += [filename]
        
        map_side_projection_labels.set(image_labels)
        map_side_projections.set(images)

@reactive.effect
@reactive.event(input.map_xyz_projection_display_size)
def update_map_xyz_projection_display_size():
    map_xyz_projection_display_size.set(input.map_xyz_projection_display_size())

@reactive.effect
@reactive.event(input.map_side_projection_display_size)
def update_map_side_projection_display_size():
    map_side_projection_display_size.set(input.map_side_projection_display_size())
