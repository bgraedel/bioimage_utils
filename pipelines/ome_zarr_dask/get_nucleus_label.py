import skimage
import edt
import dask
from csbdeep.utils import normalize
import numpy as np
from global_conf_model import modelStar
from global_conf_variable import axis_norm, scaling_factor


@dask.delayed
def nucleus_segmentation(volume: np.array, binary_mask: np.array):
    frame_h2b = volume
    img_norm = normalize(frame_h2b, 1, 99.8, axis=axis_norm)
    label, _ = modelStar.predict_instances(img_norm, prob_thresh=0.47)

    label = label * binary_mask
    return label.astype(np.uint32)


def get_cytoplasm_label(label: np.array):
    cyto_label = skimage.segmentation.expand_labels(
        label, distance=4 * scaling_factor
    ) - skimage.segmentation.expand_labels(label, distance=1)
    return cyto_label


def get_nuclear_envelope(label: np.array):
    distances = edt.edt(label)
    distance_env = 4 * scaling_factor
    nuc_env = skimage.segmentation.expand_labels(label, 2 * scaling_factor) - label * (
        distances >= distance_env
    )
    return nuc_env
