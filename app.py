from pathlib import Path
import numpy as np
import pandas as pd

from shinywidgets import render_plotly

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
selected_images_original = reactive.value([])
selected_images_rotated = reactive.value([])
selected_image_diameter = reactive.value(0)
selected_images_rotated_cropped = reactive.value([])
selected_images_title = reactive.value("Selected image:")
selected_images_labels = reactive.value([])

selected_image_diameter = reactive.value(0)

emdb_df_original = reactive.value(None)
emdb_df = reactive.value(None)

maps = reactive.value([])
map_xyz_projections = reactive.value([])
map_xyz_projection_title = reactive.value("Map XYZ projections:")
map_xyz_projection_labels = reactive.value([])
map_xyz_projection_display_size = reactive.value(128)

map_side_projections_with_alignments = reactive.value([])
map_side_projections_displayed = reactive.value([])
map_side_projection_title = reactive.value("Map side projections:")
map_side_projection_labels = reactive.value([])
map_side_projection_vertical_display_size = reactive.value(128)
 
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

            with ui.div(id="image-selection", style="max-height: 80vh; overflow-y: auto; display: flex; flex-direction: column; align-items: center;"):
                helicon.shiny.image_select(
                    id="select_image",
                    label=displayed_image_title,
                    images=displayed_images,
                    image_labels=displayed_image_labels,
                    image_size=reactive.value(128),
                    initial_selected_indices=initial_selected_image_indices,
                    allow_multiple_selection=False
                )

                @render.ui
                @reactive.event(input.show_gallery_print_button)
                def generate_ui_print_input_images():
                    req(input.show_gallery_print_button())
                    return ui.input_action_button(
                            "print_input_images",
                            "Print input images",
                            onclick=""" 
                                        var w = window.open();
                                        w.document.write(document.head.outerHTML);
                                        var printContents = document.getElementById('select_image-show_image_gallery').innerHTML;
                                        w.document.write(printContents);
                                        w.document.write('<script type="text/javascript">window.onload = function() { window.print(); w.close();};</script>');
                                        w.document.close();
                                        w.focus();
                                    """,
                            width="200px"
                        )
                
        with ui.nav_panel("Input 3D Maps"):
            with ui.div(id="input_map_files", style="display: flex; flex-direction: column; align-items: flex-start;"):
                with ui.div(style="display: flex; flex-direction: row; justify-content: space-between; align-items: flex-end; align-items: center;"):
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
                    ui.remove_ui(selector="#select_all_entries")

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
                        df["resolution"] = df["resolution"].astype(float)
                        df["twist"] = df["twist"].astype(float)
                        df["rise"] = df["rise"].astype(float)
                        df = df[cols].round(3)
                        df["rank"] = np.nan
                        df = df[["rank"] + cols]
                        emdb_df_original.set(df)

                    return ret
                
                with ui.panel_conditional("input.input_mode_maps === 'amyloid_atlas' || input.input_mode_maps === 'EMDB-helical' || input.input_mode_maps === 'EMDB'"):
                    @render.data_frame
                    @reactive.event(emdb_df, input.show_pdb, input.show_twist_star)
                    def display_emdb_dataframe():
                        ui.remove_ui(selector="#select_all_entries")
                        ui.insert_ui(
                            selector="#input_mode_maps",
                            ui=ui.input_action_button(
                                "select_all_entries",
                                label="Select all entries",
                                width="150px"
                            ),
                            where="afterEnd"
                        )
                        df = emdb_df()
                        if not input.show_pdb():
                            cols = [col for col in df.columns if col != "pdb"]
                            df = df[cols]
                        if input.show_twist_star and "twist" in df.columns and "rise" in df.columns:
                            rise = df["rise"].astype(float).abs()
                            twist_star = df["twist"].astype(float).abs()
                            # 2sub1
                            mask = (rise * 2 < 5) & (4.5 < rise * 2) & ((360 - twist_star * 2) < 90)
                            mask |= (rise < 5) & (4.5 < rise) & (abs(360 - twist_star * 2) < 90)
                            twist_star = twist_star.copy()
                            twist_star[mask] = abs(360 - twist_star[mask] * 2)
                            #3sub1
                            mask = (rise * 3 < 5) & (4.5 < rise * 3) & (abs(360 - twist_star * 3) < 90)
                            twist_star[mask] = abs(360 - twist_star[mask] * 3)                        
                            cols = df.columns.tolist()
                            twist_index = cols.index('twist')
                            cols.insert(twist_index, 'twist*')
                            df['twist*'] = np.round(twist_star, 3)                            
                            df = df[cols]
                        return render.DataGrid(
                            df,
                            selection_mode="rows",
                            filters=True,
                            editable=True,
                            height="40vh",
                            width="100%"
                        )
                
            with ui.div(style="max-height: 80vh; overflow-y: auto; display: flex; flex-direction: column; align-items: center;"):
                helicon.shiny.image_select(
                    id="display_map_xyz_projections",
                    label=map_xyz_projection_title,
                    images=map_xyz_projections,
                    image_labels=map_xyz_projection_labels,
                    image_size=map_xyz_projection_display_size,
                    enable_selection=False
                )

                @render.ui
                @reactive.event(input.show_gallery_print_button)
                def generate_ui_print_map_xyz_projection_images():
                    req(input.show_gallery_print_button())
                    return ui.input_action_button(
                            "print_map_xyz_projection_images",
                            "Print map XYZ projection images",
                            onclick=""" 
                                        var w = window.open();
                                        w.document.write(document.head.outerHTML);
                                        var printContents = document.getElementById('display_map_xyz_projections-show_image_gallery').innerHTML;
                                        w.document.write(printContents);
                                        w.document.write('<script type="text/javascript">window.onload = function() { window.print(); w.close();};</script>');
                                        w.document.close();
                                        w.focus();
                                    """,
                            width="300px"
                        )

        with ui.nav_panel("Parameters"):
            with ui.layout_columns(
                col_widths=6, style="align-items: flex-end;"
            ):
                ui.input_checkbox(
                    "ignore_blank", "Ignore blank input images", value=True
                )

                ui.input_checkbox(
                    "show_pdb", "Show PDB ids in EMDB table", value=False
                )

                with ui.tooltip(id="show_curated_helical_parameters_tooltip"):
                    ui.input_checkbox(
                        "use_curated_helical_parameters", "Use curated helical parameters", value=True
                    )

                    "When checked, the helical parameters will be updated using the curated values available at https://github.com/jianglab/EMDB_helical_parameter_curation"

                with ui.tooltip(id="show_twist_star_tooltip"):
                    ui.input_checkbox("show_twist_star", "Show twist* in EMDB table", value=True)
                    
                    "When checked, displays an additional 'twist*' column in the EMDB table. This column shows the twist angle adjusted for helical symmetry, where twist* = 360-abs(twist)*2 when 360-abs(twist)*2 < 90° and 4.5Å < rise*2 < 5Å"

            with ui.layout_columns(
                col_widths=6, style="align-items: flex-end;"
            ):
                ui.input_checkbox_group(
                    "map_projection_xyz_choices",
                    "Show projections along:",
                    choices=['x', 'y', 'z'],
                    selected=['x', 'y', 'z'],
                    inline=True
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
                ui.input_numeric(
                    "scale_range",
                    "Search image scale (percent)",
                    min=0,
                    max=100,
                    value=5,
                    step=1,
                )

            with ui.layout_columns(
                col_widths=6, style="align-items: flex-end;"
            ):
                ui.input_checkbox(
                    "rescale_apix",
                    "Resample to image pixel size", 
                    value=True
                )
                ui.input_checkbox(
                    "match_sf", 
                    "Apply matched-filter", 
                    value=False
                )
                ui.input_checkbox(
                    "show_gallery_print_button", "Show image gallery print button", value=False
                )
                ui.input_checkbox(
                    "plot_scores", "Plot matching scores", value=True
                )


title = "HelicalProjection: compare 2D images with helical structure projections"
ui.h1(title, style="font-weight: bold;")

with ui.div(style="display: flex; flex-direction: row; align-items: flex-start; gap: 10px;"):
    helicon.shiny.image_select(
        id="display_selected_image",
        label=selected_images_title,
        images=selected_images_rotated_cropped,
        image_labels=selected_images_labels,
        image_size=map_side_projection_vertical_display_size,
        justification="left",
        enable_selection=False,
    )

    with ui.layout_columns(col_widths=4):
        ui.input_slider(
            "pre_rotation",
            "Rotation (°)",
            min=-45,
            max=45,
            value=0,
            step=0.1,
        )
        
        ui.input_slider(
            "vertical_crop_size",
            "Crop vertical dimension (pixel)",
            min=32,
            max=256,
            value=0,
            step=2,
        )
        
        ui.input_slider(
            "map_side_projection_vertical_display_size",
            "Map side projection image size (pixel)",
            min=32,
            max=1024,
            value=128,
            step=32,
        )

        ui.input_radio_buttons(
            "sort_map_side_projections_by",
            "Sort projections by",
            choices=["selection", "similarity score"],
            selected="similarity score",
            inline=True
        )

        @render.ui
        @reactive.event(maps)
        def display_action_button():
            req(len(maps()))
            return ui.input_task_button("generate_projections", label="Generate Projections")


with ui.div(style="max-height: 80vh; overflow-y: auto;"):
    with ui.div(id="div_score_plot", style="display: flex; flex-direction: row; justify-content: space-between; align-items: center;"):
        @render_plotly
        @reactive.event(input.plot_scores, maps, map_side_projections_displayed,  map_side_projections_with_alignments)
        def generate_score_plot():
            req(input.plot_scores())
            req(len(map_side_projections_displayed())>1)
            req(len(map_side_projections_with_alignments())>1)

            images_work = map_side_projections_with_alignments()
            images_work = sorted(images_work, key=lambda x: -x[4])
            
            scores = [img[4] for img in images_work]
            labels = [img[-1] for img in images_work]
            try:
                titles = [""] * len(labels)
                for li, label in enumerate(labels):
                    if label in emdb_df().emdb_id.values:
                        mask = emdb_df()["emdb_id"] ==  label
                        titles[li] = str(emdb_df().loc[mask, "title"].values[0])
            except Exception as e:
                print(e)
                titles = None
            
            import plotly.express as px
            
            fig = px.scatter(
                x=range(1, len(scores)+1),
                y=scores,
                hover_name=labels,
                hover_data=dict(titles=titles),
                labels={'x': 'Rank', 'y': 'Similarity Score'},
            )
            
            fig.update_traces(
                hovertemplate='<b>%{hovertext}</b><br><i>%{customdata}</i><br>Score: %{y:.3f}<br>Rank: %{x}'
            )
            
            if len(labels) > 0:
                fig.add_annotation(
                    x=1,
                    y=scores[0],
                    text=labels[0],
                    yanchor='middle',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="black",
                    ax=70,
                    ay=0,
                    standoff=5
                )

            fig.update_layout(
                xaxis_title='Rank',
                yaxis_title='Similarity Score',
                showlegend=False,
                autosize=True,
                width=None
            )
            
            return fig

        @reactive.effect
        @reactive.event(map_side_projections_with_alignments)
        def generate_ui_select_top_n():
            ui.remove_ui(selector="#div_select_top_n")
            req(len(map_side_projections_with_alignments())>1)
            selector_ui = ui.div( 
                ui.input_numeric(
                    "select_top_n",
                    "Number of top matches:",
                    min=0,
                    value=10,
                    width="150px"
                ),

                ui.input_action_button(
                    "select_top_n_button",
                    "Select"
                ),

                id="div_select_top_n"

            )
            ui.remove_ui(selector="#div_select_top_n")
            ui.insert_ui(
                selector="#div_score_plot",
                ui=selector_ui,
                where="beforeEnd"
            )
    
    helicon.shiny.image_select(
        id="display_map_side_projections",
        label=map_side_projection_title,
        images=map_side_projections_displayed,
        image_labels=map_side_projection_labels,
        image_size=map_side_projection_vertical_display_size,
        justification="left",
        enable_selection=False
    )


    @render.ui
    @reactive.event(input.show_gallery_print_button)
    def generate_ui_print_map_side_projection_images():
        req(input.show_gallery_print_button())
        return ui.input_action_button(
                "print_map_side_projection_images",
                "Print map side projection images",
                onclick=""" 
                            var w = window.open();
                            w.document.write(document.head.outerHTML);
                            var printContents = document.getElementById('display_map_side_projections-show_image_gallery').innerHTML;
                            w.document.write(printContents);
                            w.document.write('<script type="text/javascript">window.onload = function() { window.print(); w.close();};</script>');
                            w.document.close();
                            w.focus();
                        """
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
        return
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
        return
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
@reactive.event(selected_images_original)
def update_selecte_image_rotation():
    req(len(selected_images_original()))
    rotations = [compute.estimate_rotation(img) for img in selected_images_original()]
    rotation = np.mean(rotations)
    ui.update_numeric("pre_rotation", value=round(rotation, 1))


@reactive.effect
@reactive.event(selected_images_rotated)
def update_selecte_image_diameter():
    req(len(selected_images_rotated()))
    ny =  max([img.shape[0] for img in selected_images_rotated()])
    diameter = max([compute.estimate_diameter(data=img) for img in selected_images_rotated()])
    crop_size = int(diameter * 2 + 2)//4*4
    selected_image_diameter.set(diameter)
    ui.update_numeric("vertical_crop_size", value=crop_size, max=ny)


@reactive.effect
@reactive.event(input.select_image)
def update_selecte_images_orignal():
    selected_images_original.set(
        [displayed_images()[i] for i in input.select_image()]
    )
    selected_images_labels.set(
        [displayed_image_labels()[i] for i in input.select_image()]
    )


@reactive.effect
@reactive.event(selected_images_original, input.pre_rotation)
def rotate_selected_images():
    req(len(selected_images_original()))
    if input.pre_rotation!=0:
        from skimage.transform import rotate
        rotated = []
        for img in selected_images_original():
            ny, nx = img.shape
            rotated.append(rotate(img.copy(), input.pre_rotation()))
    else:
        rotated = selected_images_original()
    selected_images_rotated.set(rotated)


@reactive.effect
@reactive.event(selected_images_rotated, input.vertical_crop_size)
def rotate_crop_selected_images():
    req(len(selected_images_rotated()))
    req(input.vertical_crop_size()>0)
    if input.vertical_crop_size()<32:
        selected_images_rotated_cropped.set(selected_images_rotated)
    else:
        d = input.vertical_crop_size()
        cropped = []
        for img in selected_images_rotated():
            ny, nx = img.shape
            if d<ny:
                cropped.append(helicon.crop_center(img, shape=(d, nx)))
            else:
                cropped.append(img)
        selected_images_rotated_cropped.set(cropped)


@reactive.effect
@reactive.event(input.input_mode_maps, input.upload_map, input.twist, input.rise, input.csym)
def get_map_from_upload():
    req(input.input_mode_maps() == "upload")
    fileinfo = input.upload_map()
    req(fileinfo)
    map_file = fileinfo[0]["datapath"]
    map_info = compute.MapInfo(filename=map_file, twist=input.twist(), rise=input.rise(), csym=input.csym(), label=fileinfo[0]["name"])
    maps.set([map_info])


@reactive.effect
@reactive.event(input.input_mode_maps, input.url_map, input.twist, input.rise, input.csym)
def get_map_from_url():
    req(input.input_mode_maps() == "url")
    req(len(input.url_map()) > 0)
    url = input.url_map()
    label = url.split("/")[-1].split(".")[0]
    map_info = compute.MapInfo(url=url, twist=input.twist(), rise=input.rise(), csym=input.csym(), label=label)
    maps.set([map_info])


@reactive.effect
@reactive.event(emdb_df_original, input.use_curated_helical_parameters)
def update_helical_parameters():
    req(emdb_df_original() is not None)
    if input.use_curated_helical_parameters():
        url = "https://raw.githubusercontent.com/jianglab/EMDB_helical_parameter_curation/refs/heads/main/EMDB_validation.csv"
        try:
            df_original = emdb_df_original()
            df_curated = pd.read_csv(url)
            df_curated = df_curated[df_curated['emdb_id'].isin(df_original['emdb_id'])]
            df_curated = df_curated[['emdb_id', 'curated_twist (°)', 'curated_rise (Å)', 'curated_csym']]
            df_updated = df_original.copy()
            df_updated = df_updated.merge(
                df_curated,
                on='emdb_id',
                how='left',
                suffixes=('', '_curated')
            )
            df_updated['twist'] = df_updated['curated_twist (°)'].combine_first(df_updated['twist']).astype(float).round(3)
            df_updated['rise'] = df_updated['curated_rise (Å)'].combine_first(df_updated['rise']).astype(float).round(3)
            df_updated['csym'] = df_updated['curated_csym'].combine_first(df_updated['csym'])
            df_updated = df_updated[df_original.columns]
            emdb_df.set(df_updated)
        except Exception as e:
             print(str(e))
             emdb_df.set(emdb_df_original())
    else:
        emdb_df.set(emdb_df_original())


@reactive.effect
def get_map_from_emdb():
    selected_indices = list(display_emdb_dataframe.cell_selection()["rows"])
    req(len(selected_indices))
    emdb_df_selected = display_emdb_dataframe.data().iloc[selected_indices]
    emdb_df_selected = emdb_df_selected.sort_values(by=display_emdb_dataframe.data_view().columns.tolist())
    maps_tmp = []
    for i in range(len(emdb_df_selected)):
        row = emdb_df_selected.iloc[i]
        emdb_id = row['emdb_id']
        twist = row['twist'] if 'twist' in row and not pd.isna(row['twist']) else 0
        rise = row['rise'] if 'rise' in row and not pd.isna(row['rise']) else 0
        csym = int(row['csym'][1:]) if 'csym' in row and not pd.isna(row['csym']) else 1
        map_info = compute.MapInfo(emd_id=emdb_id, twist=twist, rise=rise, csym=csym, label=emdb_id)
        maps_tmp.append(map_info)
    maps.set(maps_tmp)


@reactive.effect
@reactive.event(maps, input.length_z, input.map_projection_xyz_choices)
def get_map_xyz_projections():
    req(len(maps()))
    map_xyz_projections.set([])
    images = []
    image_labels = []

    with ui.Progress(min=0, max=len(maps())) as p:
        p.set(message="Generating x/yz/ projections", detail="This may take a while ...")
        
        for mi, m in enumerate(maps()):
            p.set(mi, message=f"{mi+1}/{len(maps())}: x/y/z projecting {m.label}")
            
            tmp_images, tmp_image_labels = compute.get_one_map_xyz_projects(map_info=m, length_z=input.length_z(), map_projection_xyz_choices=input.map_projection_xyz_choices())
            images += tmp_images
            image_labels += tmp_image_labels

            map_xyz_projection_labels.set(image_labels)
            map_xyz_projections.set(images)

 
@reactive.effect
@reactive.event(input.generate_projections)
def get_map_side_projections():
    req(len(maps()))
    req(len(selected_images_rotated_cropped()))
    image_query = selected_images_rotated_cropped()[0]
    image_query_label = selected_images_labels()[0]
    image_query_apix = image_apix()
    rescale_apix = input.rescale_apix()
    length_xy_factor = input.length_xy()
    match_sf = input.match_sf()
    scale_range = input.scale_range()/100

    ny, nx = image_query.shape
    arc = np.sqrt((nx/2*0.8)**2 + selected_image_diameter()**2/4)
    angle_range = min(2, round(90 - np.rad2deg(np.arccos(ny/2/arc)), 1))
    
    images = []
    with ui.Progress(min=0, max=len(maps())) as p:
        p.set(message="Generating side projections", detail="This may take a while ...")

        twist_zeros = []
        failed = []
        for mi, m in enumerate(maps()):
            if abs(m.twist) == 0:
                twist_zeros.append(m.label)
                continue

            p.set(mi, message=f"{mi+1}/{len(maps())}: symmetrizing/projecting/matching {m.label}")

            result = compute.symmetrize_project_align_one_map(m, image_query, image_query_label, image_query_apix, rescale_apix, length_xy_factor, match_sf, angle_range, scale_range)

            if result is None:
                failed.append(m.label)
                continue

            images.append(result)

    if twist_zeros:
        m = ui.modal(
            f"WARNING: twist=0°. Please set twist to a correct value for {' '.join(twist_zeros)}",
            title="Twist value error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)

    if failed:
        m = ui.modal(
            f"WARNING: failed to generate side projection of {' '.join(failed)}",
            title="Projection error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)
    
    map_side_projections_with_alignments.set(images)


@reactive.effect
@reactive.event(map_side_projections_with_alignments, input.sort_map_side_projections_by)
def update_map_side_projections_displayed():
    req(len(map_side_projections_with_alignments()))
    images_work = map_side_projections_with_alignments()
    if input.sort_map_side_projections_by() == "similarity score":
        images_work = sorted(images_work, key=lambda x: -x[4])

    df = emdb_df()
    if df is not None:
        df["rank"] = np.nan
    
    images_displayed = []
    images_displayed_labels = []
    for i, image in enumerate(images_work):
        flip, scale, rotation_angle, shift_cartesian, similarity_score, aligned_image_moving, image_query_label, proj, proj_label = image
        if df is not None and proj_label in df["emdb_id"].values:
            row_index = df.index[df["emdb_id"] == proj_label][0]
            title = str(df["title"].iloc[row_index])
            df.loc[row_index, "rank"] = i+1
        else:
            title = ""
        scale = round(scale, 3)
        rotation_angle = round(rotation_angle, 1)
        images_displayed.append(aligned_image_moving)
        images_displayed_labels.append(f"{image_query_label}:{' vflip' if flip else ''}{' '+str(scale) if scale!=1 else ''}{' '+str(rotation_angle)}°")
        images_displayed.append(proj)
        images_displayed_labels.append(f"{proj_label}: score={similarity_score:.3f}{' '+title if title else ''}")
    emdb_df.set(df.copy())
    map_side_projections_displayed.set(images_displayed)
    map_side_projection_labels.set(images_displayed_labels)


@reactive.effect
@reactive.event(input.map_xyz_projection_display_size)
def update_map_xyz_projection_display_size():
    map_xyz_projection_display_size.set(input.map_xyz_projection_display_size())

@reactive.effect
@reactive.event(input.map_side_projection_vertical_display_size)
def update_map_side_projection_vertical_display_size():
    map_side_projection_vertical_display_size.set(input.map_side_projection_vertical_display_size())


@reactive.effect
@reactive.event(input.select_all_entries)
async def select_all_entries():
    req(len(emdb_df()))
    df = display_emdb_dataframe.data_view()
    row_indces = tuple(range(len(df)))
    cols = tuple([i for i in range(len(df.columns))])
    # {'type': 'row', 'rows': (0, 1), 'cols': (0, 1, 2, 3, 4, 5, 6)}
    selection = dict(type="row", rows=row_indces, cols=cols)
    await display_emdb_dataframe.update_cell_selection(selection)


@reactive.effect
@reactive.event(input.select_top_n_button)
async def update_selected_maps_from_score_plot():
    req(len(map_side_projections_with_alignments()))
    df = display_emdb_dataframe.data()
    req(len(df))
    n = input.select_top_n()
    await display_emdb_dataframe.update_sort([{"col": df.columns.get_loc("rank"), "desc": False}])
    await display_emdb_dataframe.update_filter(
        [
            {"col": df.columns.get_loc("rank"), "value": (1, n)},
        ]
    )
    df = display_emdb_dataframe.data_view()
    row_indces = tuple(range(len(df)))
    cols = tuple([i for i in range(len(df.columns))])
    selection = dict(type="row", rows=row_indces, cols=cols)
    await display_emdb_dataframe.update_cell_selection(selection)
