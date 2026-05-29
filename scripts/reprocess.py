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
from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile, TemporaryDirectory
import subprocess

from dask.diagnostics import ProgressBar
from kerchunk.combine import MultiZarrToZarr
from kerchunk.hdf import SingleHdf5ToZarr
from msgspec import json
import earthaccess
import fsspec
import xarray as xr
import requests

from core import args, storage, storage_path


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
    with TemporaryDirectory() as tmpdir:
        tmpdir = tmpdir if args.tmpdir else "tmp"
        start = datetime.now()
        paths = earthaccess.download(
            [dataset["results"].item()],
            tmpdir,
            show_progress=False,
        )
        stop = datetime.now()
        storage.put_file(paths[0], dst_path)
    dataset["time"][...] = stop - start
    return dataset


# %% [markdown]
# #### rechunk

# %%
def process_rechunk(dataset):
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
    src_path = storage_path(dataset, chunks="")
    if not storage.exists(src_path):
        return dataset
    chunk_size = dataset["chunks"].item()
    with NamedTemporaryFile(suffix=".nc") as src:
        with NamedTemporaryFile(suffix=".nc") as dst:
            storage.get_file(src_path, src.name)
            start = datetime.now()
            subprocess.run(["nccopy", "-w", "-c", chunk_size, src.name, dst.name])
            stop = datetime.now()
            storage.put_file(dst.name, dst_path)
    dataset["time"][...] = stop - start
    return dataset


# %% [markdown]
# #### repack

# %%
def process_repack(dataset):
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
    src_path = storage_path(dataset, layout="")
    if not storage.exists(src_path):
        return dataset
    page_size = dataset["layout"].item()
    with NamedTemporaryFile(suffix=".nc") as src:
        with NamedTemporaryFile(suffix=".nc") as dst:
            storage.get_file(src_path, src.name)
            start = datetime.now()
            subprocess.run(
                ["h5repack", "-P1", "-SPAGE", f"-G{page_size}", src.name, dst.name],
            )
            stop = datetime.now()
            storage.put_file(dst.name, dst_path)
    dataset["time"][...] = stop - start
    return dataset


# %% [markdown]
# #### kerchunk

# %%
def process_single_kerchunk(dataset):
    """Kerchunk a single file in storage and push sidecar to storage.

    Execute kerchunking on a single object in storage,
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
    src_path = storage_path(dataset, virtual="")
    if not storage.exists(src_path):
        return dataset
    start = datetime.now()
    with storage.open(src_path) as src:
        reference = SingleHdf5ToZarr(src, str(src_path)).translate()
    with storage.open(dst_path, "wb") as dst:
        dst.write(json.encode(reference))
    stop = datetime.now()
    dataset["time"][...] = stop - start
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
    start = datetime.now()
    reference = MultiZarrToZarr(
        [json.decode(storage.cat(i)) for i in src_path],
        remote_protocol=storage.protocol[0],
        concat_dims="number_of_lines",
    )
    reference = reference.translate()
    with storage.open(dst_path.with_suffix(".json"), "wb") as dst:
        dst.write(json.encode(reference))
    stop = datetime.now()
    dataset["time"][...] = stop - start
    return dataset


# %% [markdown]
# ## Reprocessing

# %% [markdown]
# ### Data & Parameters

# %% [markdown]
# Define the datat to be reprocessed, and the needed reprocessing parameters, in a dictionary.

# %%
products = {
    "PACE_OCI_L2_AOP": {
        "query": {
            "version": "3.1",
            "temporal": ("2025-05-05", "2025-05-05"),
        },
        "chunks": [],
        "layout": [f"{2**22}", f"{2**23}", f"{2**24}"],
        "virtual": ["kerchunk-json"],
    },
}

# %% [markdown]
# Build an xarray.Dataset that embeds the experimental design in its dimensions (factors) and coordinates (levels).
# The `dataset` will hold timing results for the reprocessing steps.

# %%
earthaccess.login()
dataset = []
for key, value in products.items():
    results = earthaccess.search_data(
        count=int(args.count),
        short_name=key,
        **value["query"],
    )
    ds = xr.Dataset(
        {
            "results": ("granule", results),
        },
        coords={
            "product": ("granule", [key] * len(results)),
            "granule": ("granule", [i["meta"]["concept-id"] for i in results]),
        },
    )
    da = xr.DataArray(
        float("nan"),
        coords=[
            ("chunks", ["", *value["chunks"]]),
            ("layout", ["", *value["layout"]]),
            ("virtual", ["", *value["virtual"]]),
        ],
    )
    da = da.astype("timedelta64[ns]")
    ds["time"] = da
    dataset.append(ds)
dataset = xr.concat(dataset, dim="granule", data_vars="all")
dataset

# %% [markdown]
# ### Execute

# %% [markdown]
# In each cell, a selection of the dataset is created and chunked before submitting to Dask workers for a reprocessing step.

# %%
levels = {
    "chunks": [0],
    "layout": [0],
    "virtual": [0],
}
ds = dataset.isel(levels).chunk(1)
print("process_get")
with ProgressBar():
    ds = ds.map_blocks(process_get, template=ds).compute()
dataset = xr.merge((dataset, ds), join="outer", compat="no_conflicts")

# %%
levels = {
    "layout": [0],
    "virtual": [0],
}
ds = dataset.isel(levels)
ds = ds.where(ds["time"].isnull(), drop=True).chunk(1)
if ds.sizes["chunks"]:
    print("process_rechunk")
    with ProgressBar():
        ds = ds.map_blocks(process_rechunk, template=ds).compute()
    dataset = xr.merge((dataset, ds), join="outer", compat="no_conflicts")

# %%
levels = {
    "virtual": [0],
}
ds = dataset.isel(levels)
ds = ds.where(ds["time"].isnull(), drop=True).chunk(1)
if ds.sizes["layout"]:
    print("process_repack")
    with ProgressBar():
        ds = ds.map_blocks(process_repack, template=ds).compute()
    dataset = xr.merge((dataset, ds), join="outer", compat="no_conflicts")

# %%
ds = dataset
ds = ds.where(ds["time"].isnull(), drop=True).chunk(1)
if ds.sizes["virtual"]:
    print("process_single_kerchunk")
    with ProgressBar():
        ds = ds.map_blocks(process_single_kerchunk, template=ds).compute()
    dataset = xr.merge((dataset, ds), join="outer", compat="no_conflicts")
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
# View the timing results as tables.

# %%
ds = xr.load_dataset("reprocess.nc")

# %%
df = ds["time"].groupby("product").mean().to_dataframe()
# df = ds["time"].rename({"_product": "product"}).groupby("product").mean().to_dataframe()
df["time"].dropna().dt.total_seconds().reset_index()

# %%
# df = ds["time_multi_kerchunk"].to_dataframe()
# df["time_multi_kerchunk"].dropna().dt.total_seconds().reset_index()
