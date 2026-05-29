# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Core

# %% [markdown]
# ## Setup

# %%
import argparse
import os
import pathlib

# %%
import fsspec

# %% [markdown]
# ## Command Line Arguments

# %% [markdown]
# Operational values can be supplied as needed when running the scripts (see "scripts/README.md").
#
# Default settings are intended for development and testing,
# and depend on the existence of the `SCRATCH_BUCKET` environment variable,
# which is defined on the 2i2c JupyterHub and probably not defined locally.

# %%
in_cloud = "SCRATCH_BUCKET" in os.environ
in_cloud

# %%
parser = argparse.ArgumentParser()
parser.add_argument(
    "--remote",
    default=os.environ["SCRATCH_BUCKET"] + "/cloud-LDAS" if in_cloud else "data",
    help="location of storage used for reprocessed files and copies of the original",
)
parser.add_argument(
    "--tmpdir",
    default=in_cloud,
    action="store_true",
    help="whether to use a transient temporary directory for downloads (original files)",
)
parser.add_argument(
    "--count",
    default=2,
    help="the number of files to reprocess, use '-1' for all",
)
args, _ = parser.parse_known_args()
args

# %% [markdown]
# ### File handling

# %% [markdown]
# Make an `fsspec.filesystem` for storage, depending on the `remote` argument.
# Also extract the prefix from the `remote` argument.

# %%
storage, prefix = fsspec.core.url_to_fs(args.remote)  # TODO: auto_mkdir if local
storage

# %%
if hasattr(storage, "auto_mkdir"):
    storage.auto_mkdir = True

# %%
prefix = pathlib.Path(prefix)
if int(args.count) > -1:
    prefix = prefix / str(args.count)
prefix

# %%
def storage_path(dataset, **kwargs):
    """Build a path, starting from `prefix`, for outputs.

    The array, which must have a size of one, will also have coordinates used to
    construct the output path for the processed granule. The resulting path looks
    like "prefix/<product>/<chunks>/<layout>/<virtual>/<granule>".

    Parameters
    ----------
    dataset : xarray.Dataset or xarray.DataArray
        the size-one chunk of the dataset used to distribute processing
    kwargs
        override the array's coordinate value for the given keyword

    Returns
    -------
    str
        a absolute path to use for storage of outputs
    """
    path = prefix
    for item in (
        "product",
        "chunks",
        "layout",
        "virtual",
        "granule",
    ):
        index = kwargs.get(item)
        if index is None:
            index = dataset[item].item()
        path = path / str(index)
    return path
