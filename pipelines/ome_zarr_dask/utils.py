import skimage
import numpy as np


def label_to_value(tracks, labels_stack, what):
    particles_stack = np.zeros_like(labels_stack, dtype=np.uint16)
    tracks_df_norm = tracks[["t", "label", what]].copy()
    tracks_df_norm.replace([np.inf, -np.inf], np.nan, inplace=True)
    tracks_df_norm.dropna(inplace=True)
    if tracks_df_norm[what].dtype in [np.float16, np.float32, np.float64]:
        what_values = tracks_df_norm[what].to_numpy()
        max_value = what_values.max()
        min_value = what_values.min()
        # normalize to 16bit:
        normalized_values = (
            (what_values - min_value) * 65535.0 / (max_value - min_value)
        )
        normalized_values = np.clip(normalized_values, 0, 65535).astype(np.uint16)
        tracks_df_norm["what_2"] = normalized_values
        tracks_df_norm.drop(what, axis=1, inplace=True)
        tracks_df_norm.rename(columns={"what_2": what}, inplace=True)
    elif tracks_df_norm[what].dtype in [np.uint32, np.uint64, np.int32, np.int64]:
        particles_stack = np.zeros_like(labels_stack, dtype=np.uint32)
    for frame in range(labels_stack.shape[0]):
        labels_f = np.array(labels_stack[frame, :, :])

        tracks_f = tracks_df_norm[tracks_df_norm["t"] == frame]
        from_label = tracks_f["label"].values
        to_particle = tracks_f[what].to_numpy()
        skimage.util.map_array(
            labels_f, from_label, to_particle, out=particles_stack[frame, :, :]
        )
    return particles_stack
