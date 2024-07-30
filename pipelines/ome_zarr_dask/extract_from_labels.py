import dask
import skimage
import skimage.filters
import numpy as np
import edt
import pandas as pd
from scipy import ndimage as ndi
from global_conf_variable import scaling_factor, channel_h2b, channel_oct4, channel_erk


@dask.delayed
def extract_from_labels_to_df(
    i,
    frame: np.array,
    label: np.array,
    add_new_label_info_to_df: np.array,
    distlabel: np.array,
):
    """Extracts properties from the labels and returns a dataframe"""

    ## measure nuclei properties
    df_class = skimage.measure.regionprops_table(
        label,
        intensity_image=frame[channel_h2b, :, :],
        properties=(
            "label",
            "centroid",
            "area",
            "mean_intensity",
        ),
    )
    df_class = pd.DataFrame(df_class)
    df_class["t"] = i
    df_class["t"] = df_class["t"].astype(np.uint16)
    # df_class['t_min']= df_class['t']*t_interval
    df_class["centroid-0"] = df_class["centroid-0"].astype(np.uint16)
    df_class["centroid-1"] = df_class["centroid-1"].astype(np.uint16)
    df_class.rename(columns={"centroid-0": "y", "centroid-1": "x"}, inplace=True)
    df_class.rename(columns={"mean_intensity": "mean_intensity_H2B"}, inplace=True)

    #######################################################################################
    # creating a cytoplasm label.
    cyto_label = skimage.segmentation.expand_labels(
        label, distance=4
    ) - skimage.segmentation.expand_labels(label, distance=1)

    distance = 1.5 * scaling_factor
    distances = edt.edt(label)
    shrunknuc_label = label * (distances >= distance)

    #### calculate nuclear envelope
    distance_env = 4 * scaling_factor
    nuc_env = skimage.segmentation.expand_labels(label, 2 * scaling_factor) - label * (
        distances >= distance_env
    )
    nuc_smaller = label * (distances >= 5 * scaling_factor)

    # measuring mean intensity in shrunken nuclei as well as cytoplasm

    df_Meas_nuc = (
        pd.DataFrame(
            skimage.measure.regionprops_table(
                shrunknuc_label,
                intensity_image=frame[channel_erk, :, :],
                properties=("label", "mean_intensity"),
            )
        )
        .rename(columns={"mean_intensity": "mean_intensity_ERK_nuc"})
        .set_index("label")
    )

    df_Meas_cyto = (
        pd.DataFrame(
            skimage.measure.regionprops_table(
                cyto_label,
                intensity_image=frame[channel_erk, :, :],
                properties=("label", "mean_intensity"),
            )
        )
        .rename(columns={"mean_intensity": "mean_intensity_ERK_cyto"})
        .set_index("label")
    )

    df_Meas_OCT_nuc_env = (
        pd.DataFrame(
            skimage.measure.regionprops_table(
                nuc_env,
                intensity_image=frame[channel_oct4, :, :],
                properties=("label", "mean_intensity"),
            )
        )
        .rename(columns={"mean_intensity": "mean_intensity_OCT_nuc_env"})
        .set_index("label")
    )

    df_Meas_OCT_tot_nuc = (
        pd.DataFrame(
            skimage.measure.regionprops_table(
                label,
                intensity_image=frame[channel_oct4, :, :],
                properties=("label", "mean_intensity"),
            )
        )
        .rename(columns={"mean_intensity": "mean_intensity_OCT_nuc_tot"})
        .set_index("label")
    )

    df_Meas_OCT_nuc_smaller = (
        pd.DataFrame(
            skimage.measure.regionprops_table(
                nuc_smaller,
                intensity_image=frame[channel_oct4, :, :],
                properties=("label", "mean_intensity"),
            )
        )
        .rename(columns={"mean_intensity": "mean_intensity_OCT_nuc_small"})
        .set_index("label")
    )

    df_class = df_class.set_index("label", drop=False)
    df_class = df_class.join(
        [
            df_Meas_nuc,
            df_Meas_cyto,
            df_Meas_OCT_nuc_env,
            df_Meas_OCT_tot_nuc,
            df_Meas_OCT_nuc_smaller,
        ]
    )

    df_class["CNr"] = (
        df_class["mean_intensity_ERK_cyto"] / df_class["mean_intensity_ERK_nuc"]
    )
    df_class["CNr"] = df_class["CNr"].astype(np.float16)
    df_class["Oct4Ex"] = (
        df_class["mean_intensity_OCT_nuc_env"]
        / df_class["mean_intensity_OCT_nuc_small"]
    )
    df_class["Oct4Ex"] = df_class["Oct4Ex"].astype(np.float16)
    df_class["Oct4Ex_H2B_ratio"] = (
        df_class["mean_intensity_OCT_nuc_env"] / df_class["mean_intensity_H2B"]
    )
    df_class["Oct4Ex_H2B_ratio"] = df_class["Oct4Ex_H2B_ratio"].astype(np.float16)

    ### add distance to edge
    x = df_class["x"].to_numpy().astype(np.uint16)
    y = df_class["y"].to_numpy().astype(np.uint16)
    dist_t = ndi.distance_transform_edt(edge == 0)
    df_class["dist_to_edge"] = dist_t[y, x]
    df_class["dist_to_edge_label"] = distlabel[y, x]

    return df_class


@dask.delayed
def add_distance_to_edge(tracksdf, edge, distlabel):
    tracksdf = tracksdf.copy()
    x = tracksdf["x"].to_numpy().astype(np.uint16)
    y = tracksdf["y"].to_numpy().astype(np.uint16)
    dist_t = ndi.distance_transform_edt(edge == 0)
    tracksdf.loc[:, "dist_to_edge"] = dist_t[y, x]
    tracksdf.loc[:, "dist_to_edge_label"] = distlabel[y, x]
    return tracksdf


@dask.delayed
def add_new_label_info_to_df(tracksdf, label, name):
    tracksdf = tracksdf.copy()
    x = tracksdf["x"].to_numpy().astype(np.uint16)
    y = tracksdf["y"].to_numpy().astype(np.uint16)
    tracksdf.loc[:, name] = label[y, x]
    return tracksdf
