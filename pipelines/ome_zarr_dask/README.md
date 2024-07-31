# OME-ZARR Dask pipeline

This pipeline is designed to convert ND2 files into the OME-ZARR format. Additionally, it includes functionality for tracking cells and generating useful masks, such as a colony mask that delineates the borders of cell colonies. The pipeline also demonstrates how to efficiently leverage Dask's distributed computing capabilities to accelerate calculations by deploying worker nodes on a SLURM cluster.


## Usage
On your login node, you can run the pipeline with the following command:
```bash
python 00_baseline_epi_h2b_erk_oct4.py $INPUT_PATH $FOV_TO_PROCESS $1_IF_ONLY_ONE_ND2_0_IF_ALREADY_SPLIT $NUMBER_OF_JOBS_SPAWNED_ON_THE_CLUSTER $PORT_FOR_DASK_DASHBOARD
```
where:
- `$INPUT_PATH` is the path to the folder containing the nd2 files
- `$FOV_TO_PROCESS` is the field of view to process
- `$1_IF_ONLY_ONE_ND2_0_IF_ALREADY_SPLIT` is 1 if you have one big nd2 file containing all field of views, 0 if the nd2 files are already split into multiple field of views. 
- `$NUMBER_OF_JOBS_SPAWNED_ON_THE_CLUSTER` is the number of jobs you want to spawn on the cluster
- `$PORT_FOR_DASK_DASHBOARD` is the port you want to use for the Dask dashboard

The data is always saved in a subfolder called 'Analysed_Data' one directory above the input path. Metadata from the nd2 files are transferred to the OME-ZARR files. 
Settings can be adjusted using the 'global_conf_model.py' (to change the underlying stardist model), and the 'global_conf_variable.py' (change channel index of microscopy pictures) files. 

# Data viewer

The pipeline also includes a data viewer using napari that allows you to visualize the results of the pipeline. To use the data viewer, run the following command:
```bash
python 10_data_viewer.py
```