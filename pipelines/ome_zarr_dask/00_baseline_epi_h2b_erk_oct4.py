import sys
import os
import pathlib
import tifffile
import skimage
import skimage.measure
import skimage.util
import numpy as np
import pandas as pd
import dask.array as da
import zarr
import ome_zarr.io as ozi
import ome_zarr.writer as ozw
import dask
import nd2


import dask.distributed
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster


def main(input_path, pos, idv_files=False):
    from global_conf_variable import channel_h2b
    from get_colony_label import binary_processing, get_edge, get_distlabel
    from get_nucleus_label import (
        nucleus_segmentation,
        get_cytoplasm_label,
        get_nuclear_envelope,
    )
    from extract_from_labels import add_distance_to_edge, extract_from_labels_to_df
    from utils import label_to_value

    print("Start analysis, idv_files: ", idv_files)
    if idv_files:
        output_path = pathlib.Path(input_path).parent / "Analysed_Data"
    else:
        output_path = pathlib.Path(os.path.dirname(input_path)).parent / "Analysed_Data"
    print("Output path: ", output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file = input_path
    if idv_files:
        file = os.path.join(input_path, f"{str(pos).zfill(2)}.nd2")
    print(f"Start analysis for {pos}")
    with nd2.ND2File(file) as raw:
        if idv_files:
            volume = (raw.to_dask())[:, :, :, :]
        else:
            volume = (raw.to_dask())[:, pos, :, :, :]
        pixel_x_size = raw.metadata.channels[0].volume.axesCalibration[1]
        pixel_y_size = raw.metadata.channels[0].volume.axesCalibration[0]
        pixel_z_size = raw.metadata.channels[0].volume.axesCalibration[2]
        raw_ome_metadata = raw.ome_metadata().to_xml()
        raw.close()

    # volume = volume[0:2, pos, :, :, :]
    t_dim = volume.shape[0]
    n_channels = volume.shape[1]
    X_dim = volume.shape[2]
    Y_dim = volume.shape[3]

    volume = volume.rechunk((1, 1, Y_dim, X_dim))

    dest = os.path.join(output_path, f"FOV_{pos}")
    if os.path.exists(dest):
        pass
    else:
        os.makedirs(dest)

    store = ozi.parse_url(dest, mode="w").store
    root = zarr.group(store=store)
    root.attrs["omero"] = {
        "channels": [
            {
                "label": "H2B",
                "active": True,
                "window": {"end": 1000, "max": 65535, "min": 0, "start": 0},
            },
            {
                "label": "OCT4",
                "active": True,
                "window": {"end": 1500, "max": 65535, "min": 0, "start": 0},
            },
            {
                "label": "ERK",
                "active": True,
                "window": {"end": 6000, "max": 65535, "min": 0, "start": 0},
            },
        ],
        "pixel_size": {
            "y": pixel_y_size,
            "x": pixel_x_size,
            "z": pixel_z_size,
        },
    }

    images_to_save = ozw.write_image(
        image=volume,
        group=root,
        axes="tcyx",
        scaler=None,
        chunks=(1, 1, Y_dim, X_dim),
        storage_options={
            "compressor": zarr.storage.Blosc(cname="zstd", clevel=5),
        },
        compute=False,
    )
    ome_xml_grp = root.create_group("OME")
    with open(os.path.join(dest, "OME", "METADATA.ome.xml"), "w") as file:
        file.write(raw_ome_metadata)
    ome_xml_grp.attrs["bioformats2raw.layout"] = 3

    print("Finish saving raw images")

    ## used for confocal data
    # binary = (
    #     volume[:, global_conf.channel_h2b, :, :]
    #     * volume[:, global_conf.channel_erk, :, :]
    # ).map_blocks(binary_processing, dtype=bool)

    ## for our mics:
    binary = (volume[:, channel_h2b, :, :] ** 2).map_blocks(
        binary_processing, dtype=bool
    )
    binary, images_to_save = dask.persist(binary, images_to_save)
    id_area_full = da.argmax(da.all(binary, axis=(1, 2))).compute()
    print("Index area full: ", id_area_full)
    if id_area_full == 0:
        print("No area is fully covered by the colony")
    else:
        print("Recalculate binary mask")
        ## confocal
        # smooth = skimage.filters.gaussian(
        #     volume[id_area_full - 1, global_conf.channel_h2b, :, :]
        #     * volume[id_area_full - 1, global_conf.channel_erk, :, :],
        #     sigma=10,
        # )
        smooth = skimage.filters.gaussian(
            volume[id_area_full - 1, channel_h2b, :, :] ** 2,
            sigma=10,
        )
        tresh = skimage.filters.threshold_triangle(smooth)
        print("Treshold: ", tresh)

        ## confocal
        # binary[id_area_full:] = (
        #     volume[id_area_full:, global_conf.channel_h2b, :, :]
        #     * volume[id_area_full:, global_conf.channel_erk, :, :]
        # ).map_blocks(binary_processing, tresh, dtype=bool)

        binary[id_area_full:] = (
            volume[id_area_full:, channel_h2b, :, :] ** 2
        ).map_blocks(binary_processing, tresh, dtype=bool)
        binary = binary.persist()

    edge = binary.map_blocks(get_edge, dtype=bool).persist()
    distlabel = binary.map_blocks(get_distlabel, dtype=np.uint8).persist()

    labels_d = [
        da.from_delayed(
            nucleus_segmentation(volume[i, channel_h2b, :, :], binary[i]),
            shape=(volume.shape[2], volume.shape[3]),
            dtype=np.uint32,
        )
        for i in range(volume.shape[0])
    ]
    labels = da.stack(labels_d, axis=0).persist()

    cytoplasm_label = labels.map_blocks(get_cytoplasm_label, dtype=np.uint32).persist()
    nuclear_envelope = labels.map_blocks(
        get_nuclear_envelope, dtype=np.uint32
    ).persist()

    labels_names = [
        "nucleus",
        "colony",
        "edge",
    ]

    save_labels = ozw.write_labels(
        labels=labels,
        group=root,
        name=labels_names[0],
        axes="tyx",
        scaler=None,
        chunks=(1, Y_dim, X_dim),
        storage_options={
            "compressor": zarr.storage.Blosc(cname="zstd", clevel=5),
        },
        metadata={"is_grayscale_label": False},
        compute=False,
    )

    save_binary = ozw.write_labels(
        labels=binary,
        group=root,
        name=labels_names[1],
        axes="tyx",
        scaler=None,
        storage_options={
            "compressor": zarr.storage.Blosc(cname="zstd", clevel=5),
            "chunks": (1, Y_dim, X_dim),
        },
        metadata={"is_grayscale_label": False},
        compute=False,
    )

    save_edge = ozw.write_labels(
        labels=edge,
        group=root,
        name=labels_names[2],
        axes="tyx",
        scaler=None,
        storage_options={
            "compressor": zarr.storage.Blosc(cname="zstd", clevel=5),
            "chunks": (1, Y_dim, X_dim),
        },
        metadata={"is_grayscale_label": False},
        compute=False,
    )

    save_nuc_env = ozw.write_labels(
        labels=nuclear_envelope,
        group=root,
        name="nuclear envelope",
        axes="tyx",
        scaler=None,
        storage_options={
            "compressor": zarr.storage.Blosc(cname="zstd", clevel=5),
            "chunks": (1, Y_dim, X_dim),
        },
        metadata={"is_grayscale_label": False},
        compute=False,
    )

    save_cytoplasm = ozw.write_labels(
        labels=cytoplasm_label,
        group=root,
        name="cytoplasm",
        axes="tyx",
        scaler=None,
        storage_options={
            "compressor": zarr.storage.Blosc(cname="zstd", clevel=5),
            "chunks": (1, Y_dim, X_dim),
        },
        metadata={"is_grayscale_label": False},
        compute=False,
    )

    df_delayed = [
        extract_from_labels_to_df(
            i, volume[i, :, :, :], labels[i], edge[i], distlabel[i]
        )
        for i in range(volume.shape[0])
    ]
    df = pd.concat((dask.compute(df_delayed))[0])

    df_datatypes = {
        "t": np.uint16,
        "x": np.uint16,
        "y": np.uint16,
        "area": np.uint32,
        "mean_intensity_H2B": np.float32,
        "mean_intensity_ERK_nuc": np.float32,
        "mean_intensity_ERK_cyto": np.float32,
        "mean_intensity_OCT_nuc_env": np.float32,
        "mean_intensity_OCT_nuc_tot": np.float32,
        "mean_intensity_OCT_nuc_small": np.float32,
        "CNr": np.float16,
        "Oct4Ex": np.float16,
        "Oct4Ex_H2B_ratio": np.float16,
        "label": np.uint32,
        "dist_to_edge": np.float16,
        "dist_to_edge_label": np.uint8,
    }

    df = df.astype(df_datatypes)
    df.to_pickle(f"{dest}_df.xz", compression="xz")

    whats = ["CNr", "Oct4Ex", "Oct4Ex_H2B_ratio"]
    label_names = ["ERK-CNr", "norm. OCT4", "norm. OCT4 to H2B"]
    grayscales = [True, True, True]
    tifffile_names = ["ERK_CNr", "OCT4", "OCT4_H2B"]

    for what, label_name, grayscale, tifffile_name in zip(
        whats, label_names, grayscales, tifffile_names
    ):
        gen_image = label_to_value(df, labels.compute(), what)
        ozw.write_labels(
            labels=gen_image,
            group=root,
            name=label_name,
            axes="tyx",
            scaler=None,
            storage_options={
                "compressor": zarr.storage.Blosc(cname="zstd", clevel=5),
                "chunks": (1, Y_dim, X_dim),
            },
            metadata={"is_grayscale_label": grayscale},
        )
        if tifffile_name is not None:
            print("Write: ", tifffile_name)
            tifffile.imwrite(
                os.path.join(output_path, f"FOV_{pos}_{tifffile_name}.tif"),
                gen_image,
                compression="lzw",
                metadata={"axes": "TYX"},
                ome=True,
            )

    dask.compute(
        *images_to_save,
        *save_labels,
        *save_binary,
        *save_edge,
        *save_nuc_env,
        *save_cytoplasm,
    )

    print("Finish saving")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"
    dask.config.set({"distributed.scheduler.worker-ttl": None})
    input_path = sys.argv[1]
    pos = int(sys.argv[2])
    idv_files = bool(int(sys.argv[3]))
    n_jobs = int(sys.argv[4])
    port = int(sys.argv[5])
    if port is None:
        port = 8788
    cluster = SLURMCluster(
        n_workers=1,
        cores=18,
        processes=2,
        memory="42GB",
        walltime="2:00:00",
        scheduler_options={"dashboard_address": f":{port}"},
        log_directory="/tmp",
        nanny=True,
    )
    cluster.scale(jobs=n_jobs)
    client = Client(cluster)
    client.amm.start()
    print(f"Start analysis for position {pos}")
    main(input_path, pos, idv_files)
    cluster.close()
    client.close()
