# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Reprocess for Cloud Optimization

# %% [markdown]
# ## Setup

# %%
import subprocess
from collections import defaultdict
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import perf_counter

import earthaccess
import h5py
import xarray
from dask.diagnostics import ProgressBar
from kerchunk.combine import MultiZarrToZarr
from kerchunk.hdf import SingleHdf5ToZarr
from msgspec import json
from numpy import ceil, log2

# %%
from core import env, metadata_size, opts, storage, storage_path


# %% [markdown]
# ## Functions

# %% [markdown]
# #### get

# %%
def process_get(dataset):
    """Copy a file from Earthdata Cloud to storage.

    Download URLs are taken from the "files" coordinate in the input array,
    and the time it takes to download and push each file is returned in the
    corresponding value of the output array having the same coordinates.

    Parameters
    ----------
    dataset : xarray.Dataset
        the size-one chunk of the dataset used to distribute processing

    Returns
    -------
    xarray.Dataset
        the processing time, with coordinates as for `dataset`
    """
    dst_path = storage_path(dataset)
    if storage.exists(dst_path):
        time = 0
    else:
        with TemporaryDirectory() as tmpdir:
            time = perf_counter()
            paths = earthaccess.download(
                [dataset["results"].item()],
                tmpdir if opts.tmpdir else "tmp",
                show_progress=False,
            )
            time = perf_counter() - time
            storage.put_file(paths[0], dst_path)
    dataset["time"][...] = time
    dataset["size"][...] = storage.info(dst_path)["size"]
    dataset["metadata"][...] = metadata_size(dst_path)
    return dataset


# %% [markdown]
# #### rechunk

# %%
def process_rechunk(dataset, subprocess_args):
    """Fetch a granule from storage, rechunk, and push back.

    Copies an object from storage into a temporary file,
    executes rechunking with `nccopy`, and moves the result to storage.


    Parameters
    ----------
    dataset : xarray.Dataset
        see `process_get`

    Returns
    -------
    xarray.Dataset
        see `process_get`
    """
    dst_path = storage_path(dataset)
    src_path = storage_path(dataset, rechunk="null")
    if storage.exists(dst_path):
        time = 0
    else:
        # prepare for visititems
        shape = {}

        def get_shape(a, b):
            if b := getattr(b, "shape", False):
                shape[a] = "x".join((str(i) for i in b))

        # run h5repack
        args = subprocess_args[dataset["product"].item()][dataset["rechunk"].item()]
        with NamedTemporaryFile(suffix=".nc") as src:
            with NamedTemporaryFile(suffix=".nc") as dst:
                storage.get_file(src_path, src.name)
                time = perf_counter()
                if isinstance(args, defaultdict):
                    with h5py.File(src.name) as file:
                        file.visititems(get_shape)
                    args = tuple(
                        f"--layout={k}:{args[k].format(v)}" for k, v in shape.items()
                    )
                subprocess.run([f"{env}h5repack", *args, src.name, dst.name])
                time = perf_counter() - time
                storage.put_file(dst.name, dst_path)
    dataset["time"][...] = time
    dataset["size"][...] = storage.info(dst_path)["size"]
    dataset["metadata"][...] = metadata_size(dst_path)
    return dataset


# %% [markdown]
# #### repack

# %%
def process_repack(dataset, subprocess_args):
    """Fetch a granule from storage, repack, and push back.

    Copies an object from storage into a temporary file,
    executes repacking with `h5repack`, and moves the result to storage.

    Parameters
    ----------
    dataset : xarray.Dataset
        see `process_get`

    Returns
    -------
    xarray.Dataset
        see `process_get`
    """
    dst_path = storage_path(dataset)
    src_path = storage_path(dataset, repack="null")
    if storage.exists(dst_path):
        time = 0
    else:
        args = subprocess_args[dataset["product"].item()][dataset["repack"].item()]
        with NamedTemporaryFile(suffix=".nc") as src:
            with NamedTemporaryFile(suffix=".nc") as dst:
                storage.get_file(src_path, src.name)
                time = perf_counter()
                if "{size}" in args[0]:
                    size = metadata_size(src.name)
                    args = (args[0].format(size=int(2 ** ceil(log2(float(size))))),)
                subprocess.run([f"{env}h5repack", *args, src.name, dst.name])
                time = perf_counter() - time
                storage.put_file(dst.name, dst_path)
    dataset["time"][...] = time
    dataset["size"][...] = storage.info(dst_path)["size"]
    dataset["metadata"][...] = metadata_size(dst_path)
    return dataset


# %% [markdown]
# #### kerchunk

# %%
def process_single_reformat(dataset, subprocess_args):
    """Extract metadata from a single file in storage and push sidecar to storage.

    Execute "kerchunking" on a single object in storage,
    and write the resulting sidecar file to storage.

    Parameters
    ----------
    dataset : xarray.Dataset
        see `process_get`

    Returns
    -------
    xarray.Dataset
        see `process_get`
    """
    dst_path = storage_path(dataset)
    src_path = storage_path(dataset, reformat="null")
    if storage.exists(dst_path):
        time = 0
    else:
        args = subprocess_args[dataset["reformat"].item()]
        time = perf_counter()
        with storage.open(src_path) as src:
            reference = SingleHdf5ToZarr(src, str(src_path)).translate()
        with storage.open(dst_path, "wb") as dst:
            dst.write(json.encode(reference))
        time = perf_counter() - time
    dataset["time"][...] = time
    dataset["size"][...] = storage.info(dst_path)["size"]
    dataset["metadata"][...] = storage.info(dst_path)["size"]
    return dataset


# %%
def process_multi_kerchunk(dataset):
    """Combine sidecar files, and push merged sidecar to storage.

    Load the sidecar files for each file from storage, merge them,
    and write the resulting sidecar file into storage.

    Parameters
    ----------
    dataset : xarray.Dataset
        the chunk of the dataset used to distribute processing with the entire
        "file" dimension in one chunk

    Returns
    -------
    xarray.Dataset
        see `process_get`
    """
    dst_path = storage_path(dataset, granule=dataset["product"].item())
    src_path = storage.glob(str(dst_path.parent / "G*"))
    if not src_path:
        return dataset
    start = perf_counter()
    reference = MultiZarrToZarr(
        [json.decode(storage.cat(i)) for i in src_path],
        remote_protocol=storage.protocol[0],
        concat_dims="number_of_lines",
    )
    reference = reference.translate()
    with storage.open(dst_path.with_suffix(".json"), "wb") as dst:
        dst.write(json.encode(reference))
    stop = perf_counter()
    dataset["time"][...] = stop - start
    dataset["size"][...] = storage.info(dst_path)["size"]
    return dataset


# %% [markdown]
# ## Reprocessing

# %% [markdown]
# ### Data & Parameters

# %% [markdown]
# Define the datat to be reprocessed, and the needed reprocessing parameters, in a dictionary.

# %%
queries = {
    "AOP": {
        "short_name": "PACE_OCI_L2_AOP",
        "version": "3.1",
        "temporal": ("2025-05-05", "2025-05-05"),
    },
    "RRS": {
        "short_name": "PACE_OCI_L3M_RRS",
        "version": "3.1",
        "temporal": ("2025-05", "2025-05"),
        "granule_name": "*.DAY.*.4km.*",
    },
}

# %%
shared_args = {
    # the no-change args
    "null": (),
    # apply for process_repack
    "page_20": ("-SPAGE", f"-G{2**20}"),
    "page_22": ("-SPAGE", f"-G{2**22}"),
    "page_24": ("-SPAGE", f"-G{2**24}"),
    "block_16": (f"--metadata_block_size={2**14}",),
    "block_20": (f"--metadata_block_size={2**20}",),
    "block_24": (f"--metadata_block_size={2**24}",),
    "block_smart": ("--metadata_block_size={size}",),
    # apply for process_reformat
    "kerchunk_json": ("kerchunk-json",),
}
args = {
    "AOP": {
        **shared_args,
        # apply for process_rechunk
        "chunk_one": ("--layout=geophysical_data/Rrs:CHUNK=337x212x43",),
        "chunk_smart": defaultdict(
            lambda: "CHUNK={}",  # CHUNK rather than CONTIG for compression
            {
                "sensor_band_parameters/wavelength_3d": "CONTIG",
                "geophysical_data/Rrs": "CHUNK=337x212x43",
                "geophysical_data/Rrs_unc": "CHUNK=337x212x43",
            },
        ),
    },
    "RRS": {
        **shared_args,
        # apply for process_rechunk
        "chunk_one": (
            "--layout=Rrs:CHUNK=216x216x43",
        ),  # Rrs 3.1 is (16, 1024, 8), type is short, size is variable
        "chunk_smart": defaultdict(
            lambda: "CHUNK={}",  # CHUNK rather than CONTIG for compression
            {
                "Rrs": "CHUNK=216x216x43",
            },
        ),
    },
}

# %% [markdown]
# Build an xarray.Dataset that embeds the experimental design in its dimensions (factors) and coordinates (levels).
# The `dataset` will hold timing results for the reprocessing steps.

# %%
earthaccess.login()
dataset = []
stats = ["time", "size", "metadata"]
for key, value in queries.items():
    results = earthaccess.search_data(
        count=int(opts.count),
        **value,
    )
    ds = xarray.Dataset(
        {
            "granule": ("granule", [i["meta"]["concept-id"] for i in results]),
            "product": ("granule", [key] * len(results)),
            "results": ("granule", results),
        }
    )
    da = xarray.DataArray(
        data=float("nan"),
        coords=[
            ("rechunk", ["null", "chunk_one", "chunk_smart"]),
            (
                "repack",
                [
                    "null",
                    "page_20",
                    "page_22",
                    "page_24",
                    "block_16",
                    "block_20",
                    "block_24",
                    "block_smart",
                ],
            ),
            ("reformat", ["null"]),
        ],
    )
    for item in stats:
        ds[item] = da.copy()
    dataset.append(ds)
dataset = xarray.concat(dataset, dim="granule", data_vars="all")
dataset

# %% [markdown]
# ### Execute

# %%
from dask.distributed import Client

# %%
client = Client(n_workers=2, threads_per_worker=2)
client

# %% [markdown]
# In each cell, a selection of the dataset is created and chunked before submitting to Dask workers for a reprocessing step.

# %%
levels = {
    "rechunk": ["null"],
    "repack": ["null"],
    "reformat": ["null"],
}
ds = dataset.sel(levels)
if ds.sizes["granule"]:
    print("process_get")
    ds = ds.chunk(1)
    with ProgressBar():
        ds = ds.map_blocks(process_get, template=ds).compute()
    dataset = xarray.merge((dataset, ds[stats]))

# %%
levels = {
    "repack": ["null"],
    "reformat": ["null"],
}
ds = dataset.sel(levels)
ds = ds.where(ds["time"].isnull(), drop=True)
if ds.sizes["rechunk"]:
    print("process_rechunk")
    ds = ds.chunk(1)
    with ProgressBar():
        ds = ds.map_blocks(process_rechunk, (args,), template=ds).compute()
    dataset = xarray.merge((dataset, ds[stats]))

# %%
levels = {
    "reformat": ["null"],
}
ds = dataset.sel(levels)
ds = ds.where(ds["time"].isnull(), drop=True)
if ds.sizes["repack"]:
    print("process_repack")
    ds = ds.chunk(1)
    with ProgressBar():
        ds = ds.map_blocks(process_repack, (args,), template=ds).compute()
    dataset = xarray.merge((dataset, ds[stats]))

# %%
ds = dataset
ds = ds.where(ds["time"].isnull(), drop=True)
if ds.sizes["reformat"]:
    print("process_single_kerchunk")
    ds = ds.chunk(1)
    with ProgressBar():
        ds = ds.map_blocks(process_single_kerchunk, (args,), template=ds).compute()
    dataset = xarray.merge((dataset, ds[stats]))
    # ds = dataset.groupby("product").first().isel({"virtual": slice(1, None)}).chunk(1)
    # print("process_multi_kerchunk")
    # with ProgressBar():
    #     ds = ds.map_blocks(process_multi_kerchunk, template=ds).compute()
    # dataset = dataset.rename({"product": "_product"})
    # dataset["time_multi_kerchunk"] = ds["time"]

# %% [markdown]
# ### Save Timing

# %% [markdown]
# Save the dataset with all timing information, but not the `earthdata.search_data` results, to a netCDF file.

# %%
dataset.drop_vars("results").to_netcdf("reprocess.nc")

# %% [markdown]
# ## Display Results
#
# View the timing and size results as tables.
# Display the file layout.

# %%
ds = xarray.load_dataset("reprocess.nc")

# %%
reduced = {}
gb = ds.groupby(ds["product"])
reduced["avg"] = gb.mean("granule")
reduced["std"] = gb.std("granule")

# %%
reduced = [v.rename({i: f"{i}_{k}" for i in v}) for k, v in reduced.items()]
ds = xarray.merge(reduced)
df = ds.to_dataframe()

# %%
df

# %%
df.plot.bar(y="time_avg")

# %%
