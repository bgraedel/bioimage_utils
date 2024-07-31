import dask.array as da
import skimage
import skimage.filters
import numpy as np

import scipy.ndimage
from global_conf_variable import scaling_factor


def binary_processing(image: np.array, threshold: float = None) -> np.array:
    """
    Calculate the binary mask of the colony from the image

    Parameters
    ----------
    image : numpy array
        The image to be processed
    threshold : float, optional
        The threshold to be used for the binary mask, by default None, as it will be calculated automatically

    Returns
    -------
    numpy array
        The binary mask of the colony. If colony is over 60% of the image, the whole image will be returned as True
    """
    x_dim = image.shape[1]
    y_dim = image.shape[2]
    tot_pixels = x_dim * y_dim
    smooth = skimage.filters.gaussian((image), sigma=10)
    if threshold:
        tresh = threshold
    else:
        tresh = skimage.filters.threshold_triangle(smooth)

    binary = smooth > tresh
    # return binary
    area_covered = da.sum(binary.flatten()).compute() / tot_pixels
    if threshold is None and area_covered > 0.6:
        print("Over 60 of the image is covered by the colony")
        binary = np.ones_like(binary, dtype=bool)
        return binary

    print("Colony is not over 60 of the image")
    binary = skimage.morphology.remove_small_objects(binary, min_size=10000)

    binary = scipy.ndimage.binary_fill_holes(binary)
    # binary = skimage.morphology.remove_small_holes(binary, area_threshold=10000)
    binary = skimage.morphology.remove_small_holes(
        binary, area_threshold=x_dim * y_dim * 0.025
    )

    structure_dilation = skimage.morphology.disk(15)
    structure_dilation = np.expand_dims(structure_dilation, axis=0)
    binary = scipy.ndimage.binary_dilation(binary, structure=structure_dilation)
    binary = skimage.morphology.remove_small_holes(
        binary, area_threshold=x_dim * y_dim * 0.05
    )
    remove_upper_pixels = int(y_dim * 0.0217)
    remove_lower_pixel = 50
    remove_left_right_pixel = 15
    binary[0, 0:remove_upper_pixels, :] = False
    binary[0, y_dim : y_dim - remove_lower_pixel, :] = False
    binary[0, :, 0:remove_left_right_pixel] = False
    binary[0, :, x_dim : x_dim - remove_left_right_pixel] = False
    return binary


def get_edge(binary: np.array):
    """
    Calculate the edge of the binary mask

    Parameters
    ----------
    binary : numpy array
        The binary mask of the colony

    Returns
    -------
    numpy array
        The edge of the binary mask
    """
    mask = skimage.morphology.binary_erosion(binary)
    edge = binary & (~mask)
    return edge


def get_distlabel(binary: np.array):
    """
    Calculate the distance label of the binary mask (overlay that shows how far each pixel is from the edge)

    Parameters
    ----------
    binary : numpy array
        The binary mask of the colony

    Returns
    -------
    numpy array
        The distance label of the binary mask
    """

    mask = skimage.morphology.binary_erosion(binary)
    edge = binary & (~mask)

    disttrans = scipy.ndimage.distance_transform_edt(edge == 0)
    distlabel = disttrans.copy()
    label = 1
    d = 0
    while d <= disttrans.max():

        distlabel[(disttrans > d) & (disttrans <= (d + 50 * scaling_factor))] = label
        d = d + 50 * scaling_factor
        label = label + 1
        distlabel = distlabel.astype(np.uint8)
    distlabel[~binary] = 0
    disttrans[~binary] = 0
    return distlabel
