import numpy as np
import pandas as pd
from skimage.util import map_array
from utils.arrayfunctions import split_array_into_groups


def project_image(image: np.ndarray, axis: int, scale: int = 1) -> np.ndarray:
    """
    Projects image along a specified axis.

    Arguments:
    ---------
        image (np.ndarray): A NumPy array of shape (t, y, x),
            representing the label image.
        axis (int): The axis along which to project the image,
        (0 for time, 1 for y, 2 for x).
        scale (int): The scale factor for the projection.

    Returns:
    -------
        np.ndarray: The projected image.
    """
    if axis >= image.ndim:
        raise ValueError(f"Invalid axis: {axis}")
    projection = image.max(axis=axis)
    # projection = np.where(projection == 0, np.nan, projection)

    return np.repeat(projection, scale, axis=0)


def remap_segmentation(
    df: pd.DataFrame,
    segmentation: np.ndarray,
    timepoint_column: str = "timepoint",
    label_column: str = "label",
    measure_column: str = "ERK",
) -> np.ndarray:
    """
    Remaps labels in a segmentation based on measurement-values in a DataFrame.

    The DataFrame should have columns for timepoints, labels and measurements.

    Arguments:
    ---------
        df (pd.DataFrame): A pandas DataFrame containing the measurements.
        segmentation (np.ndarray): A 3D NumPy array of shape (t, y, x),
            representing the label image.
        timepoint_column (str): The name of the column containing the timepoints.
        label_column (str): The name of the column containing the labels.
        measure_column (str): The name of the column containing the measurements.

    Returns:
    -------
        np.ndarray: The remapped segmentation.
    """
    tracked_arr = (
        df[[timepoint_column, label_column, measure_column]]
        .sort_values(timepoint_column)
        .to_numpy()
    )
    grouped_arr = split_array_into_groups(
        tracked_arr[:, 0], tracked_arr[:, 1], tracked_arr[:, 2], return_group_by=True
    )
    out_arr = np.zeros_like(segmentation)
    for idx, (img, grp) in enumerate(zip(segmentation, grouped_arr)):
        arr_remapped = map_array(img, grp[:, 1], grp[:, 1])
        out_arr[idx] = arr_remapped
    return np.stack(out_arr)
