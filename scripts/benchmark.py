# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Time Data Access for Spatial Binning

# %% [markdown]
# ## Setup

# %%
from time import perf_counter

# %%
import xarray

# %%
from core import opts, prefix, storage, storage_path

# %% [markdown]
# ## Prepare Dataset

# %%
ds = xarray.load_dataset("reprocess.nc")
ds = ds.stack({"case": [...]})
da = ds["time"]
da[...] = float('nan')
ds = xarray.Dataset(
    {
        "product": ds["product"],
        "time_fsspec": da.copy(),
        "time_ros3": da.copy(),
        "time_bin": da.copy(),
    }
)
ds

# %% [markdown]
# ## Run benchmarking

# %%
for key, value in ds.groupby("case"):

    path = storage.unstrip_protocol(storage_path(value))

    fsspec_no_cache = {"open_kwargs": {"cache_type": "none"}}
    time = perf_counter()
    with xarray.open_datatree(path, engine="h5netcdf", **fsspec_no_cache) as dt:
        time = perf_counter() - time
        file = dt._close.__self__._manager._args[0]
        assert file.cache.name == "none"
        plist = dt._close.__self__.ds._root._h5file.id.get_access_plist()
        assert plist.get_page_buffer_size() == (0, 0, 0)
    value["time_fsspec"][...] = time
    
    ros3_no_cache = {"driver": "ros3", "driver_kwds": {"page_buf_size": "0", "rdcc_nbytes": 0}}
    time = perf_counter()
    with xarray.open_datatree(path, engine="h5netcdf", **ros3_no_cache) as dt:
        time = perf_counter() - time
        file = dt._close.__self__._manager._args[0]
        assert isinstance(file, str)
        plist = dt._close.__self__.ds._root._h5file.id.get_access_plist()
        assert plist.get_page_buffer_size() == (0, 0, 0)
    value["time_ros3"][...] = time


# %%
ds = ds.unstack()
ds["product"] = ds["product"].sel({"rechunk": "null", "repack": "null", "recast": "null"}, drop=True)
ds.to_netcdf("benchmark.nc")

# %% [markdown]
# ## Display Results

# %%
ds = xarray.load_dataset("benchmark.nc")

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
