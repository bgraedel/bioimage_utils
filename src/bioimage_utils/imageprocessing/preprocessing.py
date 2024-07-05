from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.optimize import curve_fit
from skimage.util import view_as_blocks
from tqdm.auto import tqdm


def bin_image(
    a: np.ndarray,
    blockshape: Tuple,
    bin_method: Literal["mean", "median", "var"] = "median",
) -> np.ndarray:
    """Calculates the blockwise mean, median, or variance of an array.

    Arguments
    ---------
        a (np.ndarray): Array to calculate blockwise median of.
        blockshape (Tuple): Shape of blocks to use.
        bin_method (str): Method to use for binning.
        Options are 'mean', 'median', 'var'.

    Returns
    -------
        np.ndarray: Binned image.
    """
    assert a.ndim == len(
        blockshape
    ), "blocks must have same dimensionality as the input image"
    assert not (
        np.array(a.shape) % blockshape
    ).any(), "blockshape must divide cleanly into the input image shape"

    block_view = view_as_blocks(a, blockshape)
    assert block_view.shape[a.ndim :] == blockshape
    block_axes = [*range(a.ndim, 2 * a.ndim)]
    if bin_method == "mean":
        return np.mean(block_view, axis=block_axes)
    elif bin_method == "median":
        return np.median(block_view, axis=block_axes)
    elif bin_method == "var":
        return np.var(block_view, axis=block_axes)


def correct_flatfield(
    raw_image: np.ndarray, flatfield_image: np.ndarray, darkfield_image: np.ndarray
) -> np.ndarray:
    """
    Corrects flatfield in a microscopy image using dark field correction and gain.

    see wiki: https://en.wikipedia.org/wiki/Flat-field_correction

    Arguments
    ---------
        raw_image (np.ndarray): numpy array, the raw image to correct
        flatfield_image (np.ndarray): numpy array, the flatfield image
        darkfield_image (np.ndarray): numpy array, the darkfield image

    Returns
    -------
        np.ndarray: Corrected image.
    """
    # Ensure the images are in float format to avoid division issues
    raw_image = raw_image.astype(np.float32)
    flatfield_image = flatfield_image.astype(np.float32)
    darkfield_image = darkfield_image.astype(np.float32)

    # Calculate the image-averaged value of (F - D)
    m = np.mean(flatfield_image - darkfield_image)

    # Calculate the gain
    gain = m / (flatfield_image - darkfield_image)

    # Apply the correction formula: C = (R - D) * G
    corrected_image = (raw_image - darkfield_image) * gain

    return corrected_image


# Function type for _exp
ExpFuncType = Callable[[np.ndarray, float, float], np.ndarray]

# Function type for _bi_exp
BiExpFuncType = Callable[[np.ndarray, float, float, float, float], np.ndarray]

# Union type for both function types
FuncType = Union[ExpFuncType, BiExpFuncType]


def _exp(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.exp(-b * x)


def _bi_exp(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return (a * np.exp(-b * x)) + (c * np.exp(-d * x))


def _exponential_correct(
    images: np.ndarray,
    contrast_limits: Optional[Tuple[int, int]] = None,
    method: Literal["mono", "bi"] = "mono",
) -> np.ndarray:
    """Corrects photobleaching in a 3D or 4D image stack using an exponential curve.

    Adapted from:
    https://github.com/marx-alex/napari-bleach-correct/blob/main/src/napari_bleach_correct/modules/exponential.py.
    """
    dtype = images.dtype
    if contrast_limits is None:
        contrast_limits = (np.min(images), np.max(images))
    assert (
        3 <= len(images.shape) <= 4
    ), f"Expected 3d or 4d image stack, instead got {len(images.shape)} dimensions"

    avail_methods = ["mono", "bi"]
    func: FuncType
    if method == "mono":
        func = _exp
    elif method == "bi":
        func = _bi_exp
    else:
        raise NotImplementedError(
            f"method must be one of {avail_methods}, instead got {method}"
        )

    axes = tuple(range(len(images.shape)))
    I_mean = np.mean(images, axis=axes[1:])
    x_data = np.arange(images.shape[0])

    with np.errstate(over="ignore"):
        try:
            popt, _ = curve_fit(func, x_data, I_mean)
            f_ = np.vectorize(func)(x_data, *popt)
        except (ValueError, RuntimeError, Warning):
            f_ = np.ones(x_data.shape)

    residuals = I_mean - f_
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((I_mean - np.mean(I_mean)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R^2: {r_squared}")

    f = f_ / np.max(f_)
    if len(images.shape) == 3:
        f = f.reshape(-1, 1, 1)
    else:
        f = f.reshape(-1, 1, 1, 1)
    images = images / f

    images[images < contrast_limits[0]] = contrast_limits[0]
    images[images > contrast_limits[1]] = contrast_limits[1]
    return images.astype(dtype)


def _match_histogram(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    hist_source, bin_edges = np.histogram(source.ravel(), bins=65536, range=(0, 65536))
    hist_template, _ = np.histogram(template.ravel(), bins=65536, range=(0, 65536))

    cdf_source = hist_source.cumsum() / hist_source.sum()
    cdf_template = hist_template.cumsum() / hist_template.sum()

    lookup_table = np.interp(cdf_source, cdf_template, np.arange(65536)).astype(
        np.uint16
    )

    matched = lookup_table[source.ravel()].reshape(source.shape)
    return matched


def _match_histogram_stack(img_stack: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Match the histogram of a 3D image stack to a template image."""
    matched_stack = np.zeros_like(img_stack)
    for i in tqdm(range(img_stack.shape[0])):
        matched_stack[i] = _match_histogram(img_stack[i], template)
    return matched_stack


def _simple_ratio_correct(images: np.ndarray) -> np.ndarray:
    """Corrects photobleaching in a 3D or 4D image stack using a simple ratio method."""
    dtype = images.dtype
    I_mean = np.mean(images, axis=tuple(range(1, len(images.shape))))
    ratio = I_mean[0] / I_mean

    if len(images.shape) == 3:
        ratio = ratio.reshape(-1, 1, 1)
    else:
        ratio = ratio.reshape(-1, 1, 1, 1)

    corrected_images = images * ratio
    return corrected_images.astype(dtype)


def bleach_correction(
    images: np.ndarray,
    method: Literal[
        "exponential_mono", "exponential_bi", "histogram_matching", "simple_ratio"
    ],
    contrast_limits: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Apply bleach correction to image stacks.

    Available methods are:
    exponential_mono, exponential_bi, histogram_matching, simple_ratio.

    Assumes image shape is (T, Y, X).

    Arguments
    ---------
        images (np.ndarray): Image stack to correct.
        method (str): Method to use for bleach correction.
        contrast_limits (Tuple): Contrast limits to apply to the corrected image.
            Only used for exponential correction methods.

    Returns
    -------
        np.ndarray: Corrected image stack.
    """
    if method == "exponential_mono":
        return _exponential_correct(images, contrast_limits, method="mono")
    elif method == "exponential_bi":
        return _exponential_correct(images, contrast_limits, method="bi")
    elif method == "histogram_matching":
        return _match_histogram_stack(images, images[0])
    elif method == "simple_ratio":
        return _simple_ratio_correct(images)
    else:
        raise ValueError(f"Unknown method: {method}")


def remove_image_background(
    image: np.ndarray,
    filter_type: str = "gaussian",
    size: Tuple[int, int, int] = (10, 1, 1),
    dims: str = "TXY",
    crop_time_axis: bool = False,
) -> np.ndarray:
    """Removes background from images.

    Assumes axis order (t, y, x) for 2d images and (t, z, y, x) for 3d images.

    Arguments:
    ---------
        image (np.ndarray): Image to remove background from.
        filter_type (Union[str, function]): Filter to use to remove background.
            Can be one of ['median', 'gaussian'].
        size (int, Tuple): Size of filter to use. For median filter,
            this is the size of the window.
            For gaussian filter, this is the standard deviation.
            If a single int is passed in, it is assumed to be the same for all dims.
            If a tuple is passed in, it is assumed to correspond to the size
            of the filter in each dimension.Default is (10, 1, 1).
        dims (str): Dimensions to apply filter over. Can be one of ['TXY', 'TZXY'].
            Default is 'TXY'.
        crop_time_axis (bool): Whether to crop the time axis. Default is True.

    Returns:
    -------
        np.ndarray: Corrected image.
    """
    # correct images with a filter applied over time
    allowed_filters = ["median", "gaussian"]
    dims_list = list(dims.upper())

    # check input
    for i in dims_list:
        if i not in dims_list:
            raise ValueError(f"Invalid dimension {i}. Must be 'T', 'X', 'Y', or 'Z'.")

    if len(dims_list) > len(set(dims_list)):
        raise ValueError("Duplicate dimensions in dims.")

    if len(dims_list) != image.ndim:
        raise ValueError(
            f"Length of dims must be equal to number of dimensions in image.\
            Image has {image.ndim} dimensions."
        )
    # make sure axis dont occur twice and that they are valid
    if len(dims) != len(set(dims)):
        raise ValueError("Dimensions must not occur twice.")

    if filter_type not in allowed_filters:
        raise ValueError(f"Filter type must be one of {allowed_filters}.")

    # get index of time axis
    t_idx = dims_list.index("T")

    orig_image = image.copy()

    if isinstance(size, int):
        size = (size,) * image.ndim
    elif isinstance(size, tuple):
        if len(size) != image.ndim:
            raise ValueError(f"Filter size must have {image.ndim} dimensions.")
        # check size of dimensions are compatible with image
        for idx, s in enumerate(size):
            if s > image.shape[idx]:
                raise ValueError(
                    f"Filter size in dimension {idx} is larger than\
                    image size in that dimension."
                )
    else:
        raise ValueError("Filter size must be an int or tuple.")

    if filter_type == "median":
        filtered = median_filter(orig_image, size=size)
    elif filter_type == "gaussian":
        filtered = gaussian_filter(orig_image, sigma=size)

    # crop time axis if necessary
    shift = size[t_idx] // 2
    corr = np.subtract(orig_image, filtered, dtype=np.float32)
    if crop_time_axis:
        corr = corr[shift:-shift]

    return corr
