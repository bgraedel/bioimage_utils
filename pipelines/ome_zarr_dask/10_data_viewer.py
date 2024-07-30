from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import napari
from magicgui import magicgui
import os
import pickle
import pandas as pd
import lzma
from qtpy.QtWidgets import QLabel, QSizePolicy

upper_quantile = 0.95
lower_quantile = 0.05

base_path_stem_cell = "SET_YOUR_BASE_PATH_HERE"
extra_data_folders_file = os.path.join(
    base_path_stem_cell, "DataViewer", "extra_data_folders.txt"
)  # here you can set an extra txt file with the names of the folders you want to load


def find_subfolders_with_analysed_data(directory):
    result = []
    for subfolder in os.listdir(directory):
        subfolder_path = os.path.join(directory, subfolder)
        if os.path.isdir(subfolder_path):
            analysed_data_folder = os.path.join(subfolder_path, "Analysed_Data")
            if os.path.isdir(analysed_data_folder):
                result.append(subfolder_path)
    return result


def sort_key(s):
    if s == "FOV_0":
        return -1
    return int(s.split("_")[1])


def find_fov_choices(project_path):
    base_folder = os.path.join(base_path_stem_cell, project_path)
    fovs = [
        os.path.basename(f)
        for f in os.listdir(os.path.join(base_folder, "Analysed_Data"))
        if os.path.isdir(os.path.join(base_folder, "Analysed_Data", f))
    ]
    fovs = sorted(fovs, key=sort_key)
    return fovs


# data_folders = find_subfolders_with_analysed_data(base_path_stem_cell)
# data_folder_names = [os.path.basename(folder) for folder in data_folders]
data_folder_names = []
for line in open(extra_data_folders_file, "r"):
    data_folder_names.append(line.strip())

current_fovs = find_fov_choices(data_folder_names[0])
current_fov = current_fovs[0]
project_c = data_folder_names[0]


@magicgui(
    project={
        "label": "Project: ",
        "choices": data_folder_names,
        "value": data_folder_names[0],
    },
    fov={
        "label": "Position: ",
        "choices": find_fov_choices(data_folder_names[0]),
        "value": current_fov,
    },
    next_fov={"widget_type": "PushButton", "label": "Next FOV ->"},
    previous_fov={"widget_type": "PushButton", "label": "Previous FOV <-"},
    load_tracking_data={
        "label": "Load Tracking Data",
        "widget_type": "PushButton",
        "visible": False,
    },
    call_button="Load Data",
)
def load_data_widget(
    project,
    fov,
    next_fov: bool = False,
    previous_fov: bool = False,
    load_tracking_data: bool = False,
):
    global current_fov
    global project_c
    current_fov = fov
    project_c = project
    viewer.layers.clear()
    url = os.path.join(base_path_stem_cell, project, "Analysed_Data", fov)

    reader = Reader(parse_url(url))
    nodes = list(reader())
    viewer.add_image(
        nodes[0].data,
        channel_axis=next(
            (
                i
                for i, item in enumerate(nodes[0].metadata["axes"])
                if item["name"] == "c"
            ),
            None,
        ),
        name=nodes[0].metadata["channel_names"],
        contrast_limits=nodes[0].metadata["contrast_limits"],
        visible=False,
    )

    labels = nodes[1].zarr.root_attrs["labels"]
    for i in range(2, len(nodes)):
        if nodes[i].zarr.root_attrs["multiscales"][0]["metadata"]["is_grayscale_label"]:
            viewer.add_image(nodes[i].data, name=labels[i - 2], visible=False)
        else:
            viewer.add_labels(nodes[i].data, name=labels[i - 2], visible=False)

    graph_file_path = os.path.join(
        base_path_stem_cell, project, "Analysed_Data", f"{fov}_graph_2.xz"
    )
    if os.path.exists(graph_file_path):
        load_data_widget.load_tracking_data.visible = True
    viewer.dims.set_current_step(0, 0)


@load_data_widget.load_tracking_data.changed.connect
def load_tracking_data(value):
    graph_file_path = os.path.join(
        base_path_stem_cell, project_c, "Analysed_Data", f"{current_fov}_graph_2.xz"
    )
    if os.path.exists(graph_file_path):
        with lzma.open(graph_file_path, "rb") as f:
            graph = pickle.load(f)
        tracks_df = pd.read_pickle(
            os.path.join(
                base_path_stem_cell,
                project_c,
                "Analysed_Data",
                f"{current_fov}_df_tracks_2.xz",
            ),
            compression="xz",
        )

        upper_q = tracks_df["distance"].quantile(upper_quantile)
        lower_q = tracks_df["distance"].quantile(lower_quantile)
        tracks_df["distance_clip"] = tracks_df["distance"].clip(
            lower=lower_q, upper=upper_q
        )
        distance = tracks_df["distance_clip"].values
        viewer.add_tracks(
            tracks_df[["track_id", "t", "y", "x"]].values,
            graph=graph,
            features={"distance": distance},
        )


@load_data_widget.next_fov.changed.connect
def set_next(value):
    global current_fov
    fov_choices = find_fov_choices(project_c)
    if current_fov in fov_choices:
        current_index = fov_choices.index(current_fov)
        if current_index < len(fov_choices) - 1:
            load_data_widget.fov.value = fov_choices[current_index + 1]
    elif current_fov is None:
        load_data_widget.fov.value = fov_choices[0]
    load_data_widget.call_button.clicked.emit()


@load_data_widget.previous_fov.changed.connect
def set_previous(value):
    global current_fov
    fov_choices = find_fov_choices(project_c)
    if current_fov in fov_choices:
        current_index = fov_choices.index(current_fov)
        if current_index > 0:
            load_data_widget.fov.value = fov_choices[current_index - 1]
    elif current_fov is None:
        load_data_widget.fov.value = fov_choices[0]
    load_data_widget.call_button.clicked.emit()


@load_data_widget.project.changed.connect
def on_project_change(event=None):
    fovs = find_fov_choices(load_data_widget.project.value)
    load_data_widget.fov.choices = fovs
    load_data_widget.fov._default_choices = fovs
    load_data_widget.fov.value = fovs[0] if fovs else None


viewer = napari.Viewer()
load_fov_widget = load_data_widget
load_data_widget.max_width = 500
# load_data_widget.setFixedWidth(50)
# load_fov_widget.show(run=True)
# load_fov_widget.set
load_data_block = viewer.window.add_dock_widget(load_fov_widget, name="Load Data")
if __name__ == "__main__":
    napari.run()
