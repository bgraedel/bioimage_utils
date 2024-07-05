import numpy as np


def _group_data(frame_data: np.ndarray) -> tuple:
    unique_frame_vals, unique_frame_indices = np.unique(
        frame_data, axis=0, return_index=True
    )
    return unique_frame_vals.astype(np.int32), unique_frame_indices[1:]


def split_array_into_groups(
    group_by: np.ndarray, *args: np.ndarray, return_group_by: bool = True
) -> tuple:
    """
    Split arrays into groups based on the values of the first array.

    Works similar to pandas groupby.

    Arguments
    ---------
        group_by (np.ndarray): The array to group by.
        *args (np.ndarray): The arrays to split into groups.
        return_group_by (bool): Whether to return the group_by array.

    Returns
    -------
        tuple: Sorted group_by array and the split arrays.
    """
    group_by_sort_key = np.argsort(group_by)
    group_by_sorted = group_by[group_by_sort_key]
    _, group_by_cluster_id = _group_data(group_by_sorted)

    result = [group_by_sort_key]

    if return_group_by:
        result.append(np.split(group_by_sorted, group_by_cluster_id))

    for arg in args:
        assert len(arg) == len(
            group_by
        ), "All arguments must have the same length as group_by."
        arg_sorted = arg[group_by_sort_key]
        result.append(np.split(arg_sorted, group_by_cluster_id))

    return tuple(result)
