"""
Compute and report contiguous- and chunked-dataset storage statistics for an
HDF5 file that ``h5stat`` (the stock utility shipped with the HDF5 library) does
not provide.

Overview
--------
``h5stat`` gives a good summary of metadata, free space, and raw storage, but it
stops short of the dataset-level layout details that if importance for files
that use the ``PAGE`` file space strategy and are read over the network through
the ROS3 virtual file driver. This script fills that gap by walking every
dataset in a file and building per-dataset statistics:

* For **contiguous** datasets:
    - stored size in bytes
    - whether the dataset fits inside a single file page or spills out (paged
      files only)

* For **chunked** datasets:
    - number of chunks actually stored, total stored size, min/max stored chunk
      size, nominal (uncompressed) chunk size
    - ratio of chunk extent to dataset extent
    - distribution of chunks across file pages, and a "page spread anomaly"
      counter that flags chunks scattered over more pages than their total
      stored size requires (paged files only)

Compact datasets are intentionally skipped because their storage is inlined in
object headers and is not comparable to the other two layouts.

Results are aggregated into histograms rendered as plain-text tables via
``tabulate``; individual per-dataset records can also be dumped as text or JSON
for post-processing.

Input sources
-------------
The ``h5file`` argument accepts:

* A local filesystem path.
* An ``https://`` URL --- opened with the HDF5 ROS3 VFD.
* An ``s3://`` URI --- opened with the HDF5 ROS3 VFD.

Command-line options
--------------------
``h5file``
    Required. Path, ``https://`` URL, or ``s3://`` URI of the input file.

``--page-buffer-size BYTES``
    Size of the HDF5 page buffer cache used when reopening a paged file to
    collect per-chunk page information. Default: 64 MiB. Has no effect on
    non-paged files.

``--show``
    Print one line (or one JSON object with ``--json``) per dataset instead of
    the aggregate histogram tables.

``--json``
    With ``--show``, emit the per-dataset records as a JSON array sorted by
    dataset name. Ignored without ``--show``.

``--page-list {page,dataset}``
    Only valid for paged files. Emit a JSON mapping of either
    file-page-to-datasets (``page``) or dataset-to-file-pages (``dataset``),
    separating datasets that fit inside pages from those that spill out.

Output
------
Default mode prints a series of tables covering, as applicable:

* contiguous stored-size distribution and in-page/out-of-page split
* chunked total-stored-size distribution
* chunk-size distribution
* chunk-to-dataset shape ratio distribution
* number-of-chunks distribution
* recommended chunk-cache-size distribution (``chunk_size * num_chunks``)
* for paged files: chunks-vs-page-size split, pages-per-dataset distribution,
  file-page spread anomaly distribution, and the maximum fraction of a dataset's
  chunks landing in any single file page

Runtime requirements
--------------------
* Python 3 with ``h5py``, ``numpy``, and ``tabulate``.
* HDF5 library 1.14.3 or later. Version 2.0 or later highly recommended.
* HDF5 built with the ROS3 virtual file driver when reading ``https://`` or
  ``s3://`` sources.
"""

import argparse
import json
import operator
import logging
from collections import defaultdict
from dataclasses import dataclass
from functools import partial, reduce
import os
from typing import Any, Union
from configparser import ConfigParser
from pathlib import Path

import h5py
import numpy as np
from tabulate import tabulate

logger = logging.getLogger("h5stat")

HDF5_VERSION = h5py.h5.get_libversion()
if HDF5_VERSION < (1, 14, 3):
    raise RuntimeError("Requires HDF5 library 1.14.3 or later")
elif not h5py.h5.get_config().ros3:
    raise RuntimeError("HDF5 library must be built with ROS3 virtual file driver")

MiB = 1024 * 1024


def get_cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Provide contiguous and chunked dataset statistics that h5stat does not do.",
        epilog="Developed by The HDF Group. This work was supported by NASA/GSFC under Raytheon Company contract 80GSFC21CA001.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("h5file", help="Input HDF5 file name.", type=str)
    parser.add_argument(
        "--page-buffer-size",
        help="Page buffer cache size in bytes. Default 64 MiB.",
        type=int,
        default=64 * MiB,
    )
    parser.add_argument("--show", help="Print individual dataset stats", action="store_true")
    parser.add_argument(
        "--json", help="Format individual dataset stats in JSON", action="store_true"
    )
    parser.add_argument(
        "--page-list",
        help="List datasets per their file pages in JSON. Only for paged files.",
        type=str,
        choices=("page", "dataset"),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
    )
    return parser.parse_args()


def get_s3_params(need_region: bool = False) -> dict[str, bytes]:
    """Collect AWS-like S3 connection parameters.

    Resolution order for each value is: environment variable, then the
    matching entry in the active profile of the AWS credentials/config
    files, then an empty string. ``need_region`` adds ``aws_region`` to the
    returned dict (required for ``s3://`` but not ``https://`` access).
    Values are returned as ASCII-encoded bytes because that is what the
    ROS3 driver in h5py expects.
    """
    s3p = dict()

    # Read AWS credentials and config files...
    home = Path.home()
    creds = ConfigParser()
    creds.read(os.getenv("AWS_SHARED_CREDENTIALS_FILE", home.joinpath(".aws", "credentials")))
    config = ConfigParser()
    config.read(os.getenv("AWS_CONFIG_FILE", home.joinpath(".aws", "config")))

    profile = os.getenv("AWS_PROFILE", "default")
    s3p["secret_id"] = os.getenv(
        "AWS_ACCESS_KEY_ID", creds.get(profile, "aws_access_key_id", fallback="")
    ).encode("ascii")
    s3p["secret_key"] = os.getenv(
        "AWS_SECRET_ACCESS_KEY",
        creds.get(profile, "aws_secret_access_key", fallback=""),
    ).encode("ascii")
    s3p["session_token"] = os.getenv(
        "AWS_SESSION_TOKEN",
        creds.get(profile, "aws_session_token", fallback=""),
    ).encode("ascii")
    if need_region:
        s3p["aws_region"] = os.getenv("AWS_REGION", config.get(profile, "region")).encode("ascii")

    return s3p


# ---------------------------------------------------------------------------- #
# Per-dataset record types
#
# Each dataset in the input file is reduced to exactly one ContStats or
# ChunkStats instance. They are frozen dataclasses so the list of records can
# be freely sorted and serialised without accidental mutation. Both types
# know how to emit a plain-text line (to_print) and a dict suitable for JSON
# dumping (to_dict) for the --show mode.
# ---------------------------------------------------------------------------- #

@dataclass(slots=True, frozen=True)
class ContStats:
    """Stats for one contiguous HDF5 dataset.

    ``page_bins`` is empty for non-paged files; for paged files it maps the
    (1-indexed) file page containing the dataset to a count of 1.
    ``out_of_page`` is True when the dataset's stored size exceeds the file
    page size and therefore cannot live inside a single page.
    """

    name: str
    stor_size: int
    page_bins: dict
    out_of_page: bool

    def to_dict(self) -> dict[str, Any]:
        d = {
            "dataset": self.name,
            "stored_size": self.stor_size,
        }
        if len(self.page_bins) or self.out_of_page:
            d.update(
                {
                    "file_pages": self.page_bins,
                    "out_of_page": self.out_of_page,
                }
            )
        return d

    def to_print(self) -> str:
        if len(self.page_bins):
            return (
                f"dataset={self.name} stored_size={self.stor_size}"
                f" file_pages={len(self.page_bins)} out_of_page={self.out_of_page}"
            )
        else:
            return f"dataset={self.name} stored_size={self.stor_size}"


@dataclass(slots=True, frozen=True)
class ChunkStats:
    """Stats for one chunked HDF5 dataset.

    ``size`` is the nominal (uncompressed) chunk size in bytes, computed
    from the chunk shape and datatype size; ``stor_size`` is the total
    bytes actually on disk for all stored chunks. ``extent_ratio`` is the
    product of ``min(1, chunk[i] / shape[i])`` across all dimensions and
    must lie in (0, 1]. ``page_spread_anomaly`` counts file pages beyond
    the minimum required to hold ``stor_size`` bytes --- a proxy for how
    badly the chunks are scattered in the file; it must be non-negative.
    """

    name: str
    num_stored: int
    size: int
    stor_size: int
    min_size: int
    max_size: int
    extent_ratio: float
    page_bins: dict
    page_spread_anomaly: int

    def __post_init__(self):
        if self.extent_ratio > 1:
            raise ValueError(f"Chunk shape ratio greater than 1 for {self.name}")
        if self.page_spread_anomaly < 0:
            raise ValueError(f"Chunks file page spread anomaly negative for {self.name}")

    def to_dict(self) -> dict[str, Any]:
        d = {
            "dataset": self.name,
            "chunks_stored": self.num_stored,
            "chunk_size": self.size,
            "stored_size": self.stor_size,
            "min_stored_chunk_size": self.min_size,
            "max_stored_chunk_size": self.max_size,
            "chunk_shape_ratio": self.extent_ratio,
        }
        if len(self.page_bins):
            d.update(
                {
                    "file_pages": self.page_bins,
                    "page_spread_anomaly": self.page_spread_anomaly,
                }
            )
        return d

    def to_print(self) -> str:
        if len(self.page_bins):
            return (
                f"dataset={self.name} stored_size={self.stor_size} chunks_stored={self.num_stored}"
                f" chunk_size={self.size} min_stored_chunk_size={self.min_size} max_stored_chunk_size={self.max_size}"
                f" chunk_shape_ratio={self.extent_ratio:.6g} file_pages={len(self.page_bins)}"
                f" page_spread_anomaly={self.page_spread_anomaly}"
            )
        else:
            return (
                f"dataset={self.name} stored_size={self.stor_size} chunks_stored={self.num_stored}"
                f" chunk_size={self.size} min_stored_chunk_size={self.min_size} max_stored_chunk_size={self.max_size}"
                f" chunk_shape_ratio={self.extent_ratio:.6g}"
            )


def chunk_to_shape_ratio(chunk: tuple, shape: tuple) -> float:
    """Ratio of chunk to dataset shape extent.

    Each per-dimension contribution is clamped to 1 so that chunks
    exceeding the dataset extent in a given dimension do not inflate the
    ratio beyond 1. Empty dimensions (size 0) are skipped rather than
    raising ZeroDivisionError, which can occur on unfilled 1-D datasets.
    """
    ratio = 1
    for c, s in zip(chunk, shape):
        try:
            ratio *= min(1, c / s)
        except ZeroDivisionError:
            # Deal with 1D datasets without data...
            continue
    return ratio


def chunk_info(dset: h5py.Dataset, page_size: int) -> tuple[dict[int, int], int, int]:
    """Determine file page and the smallest and largest chunk size of a chunked
    dataset.

    Iterates every stored chunk via ``chunk_iter`` and returns a tuple of
    ``(page_bins, min_stored_chunk_size, max_stored_chunk_size)``, where
    ``page_bins`` maps 1-indexed file page number to the count of chunks
    stored in that page. Chunks whose stored size already exceeds
    ``page_size`` cannot fit in a single page and are not binned. A chunk
    that is small enough to fit in a page but straddles a page boundary
    is treated as a layout error and raises ``ValueError``.

    Only meaningful for files with the ``PAGE`` file space strategy.
    """
    stinfo = defaultdict(int)
    chunk_sizes = list()

    def _record_chunk(chunk_stor) -> None:
        chunk_sizes.append(chunk_stor.size)
        if chunk_stor.size <= page_size:
            start_page = np.floor(chunk_stor.byte_offset / page_size).astype(int).item() + 1
            end_page = (
                np.floor((chunk_stor.byte_offset + chunk_stor.size - 1) / page_size)
                .astype(int)
                .item()
                + 1
            )
            if start_page != end_page:
                raise ValueError(f"Chunk crosses file page boundary: {chunk_stor}")
            stinfo[start_page] += 1

    dset.id.chunk_iter(_record_chunk)

    return stinfo, min(chunk_sizes), max(chunk_sizes)


def cont2page(dset: h5py.Dataset, page_size: int) -> dict[int, int]:
    """Determine file page of a contiguous dataset.

    Returns a dict mapping the 1-indexed file page containing the dataset
    to 1, or an empty dict when the dataset has no allocated storage or
    is too large to fit inside a single page. Raises ``ValueError`` if a
    dataset small enough to fit in one page nevertheless straddles a
    page boundary, which indicates a malformed file layout.
    """
    stinfo = defaultdict(int)
    offs = dset.id.get_offset()
    size = dset.id.get_storage_size()
    if offs is not None and size <= page_size:
        start_page = np.floor(offs / page_size).astype(int).item() + 1
        end_page = np.floor((offs + size - 1) / page_size).astype(int).item() + 1
        if start_page != end_page:
            raise ValueError(f"Contiguous dataset crosses file page boundary: {dset.name}")
        stinfo[start_page] += 1

    return stinfo


def dset_stats(
    name: str,
    h5obj: Union[h5py.Group, h5py.Dataset],
    dset_list: list[Union[ChunkStats, ContStats]],
    page_size: int = 0,
) -> None:
    """Visitor callback that appends one stats record per dataset.

    Intended for use with ``h5py.Group.visititems``. Groups and compact
    datasets are silently skipped; every other dataset produces one
    ``ChunkStats`` (chunked layout) or ``ContStats`` (contiguous layout)
    record appended to ``dset_list``. ``page_size`` of 0 disables all
    page-related calculations and is correct for non-paged files.
    """
    if isinstance(h5obj, h5py.Dataset):
        chunk_shape = h5obj.chunks
        if chunk_shape:
            chunk_nelem = reduce(operator.mul, chunk_shape, 1)
            chunk_pages, min_size, max_size = chunk_info(h5obj, page_size)
            if page_size:
                num_chunks = reduce(operator.add, chunk_pages.values(), 0)
                stored_size = h5obj.id.get_storage_size()
                if max_size > page_size:
                    page_spread = 0
                else:
                    page_spread = (
                        len(chunk_pages) - np.ceil(stored_size / page_size).astype(int).item()
                    )
            else:
                num_chunks = h5obj.id.get_num_chunks()
                stored_size = h5obj.id.get_storage_size()
                page_spread = 0
            logger.info(f"visited {stored_size} bytes chunked dataset {h5obj.name} with max chunk {max_size}")
            dset_list.append(
                ChunkStats(
                    name=h5obj.name,
                    num_stored=num_chunks,
                    extent_ratio=chunk_to_shape_ratio(chunk_shape, h5obj.shape),
                    stor_size=stored_size,
                    min_size=min_size,
                    max_size=max_size,
                    size=h5obj.id.get_type().get_size() * chunk_nelem,
                    page_bins=chunk_pages,
                    page_spread_anomaly=page_spread,
                )
            )
        else:
            if h5obj.id.get_create_plist().get_layout() == h5py.h5d.COMPACT:
                # Compact datasets are not included due to their specific storage...
                return
            stored_size = h5obj.id.get_storage_size()
            logger.info(f"visited {stored_size} bytes contiguous dataset {h5obj.name}")
            dset_list.append(
                ContStats(
                    name=h5obj.name,
                    stor_size=stored_size,
                    page_bins=cont2page(h5obj, page_size),
                    out_of_page=True if page_size and stored_size > page_size else False,
                )
            )


def stats_table(
    bin_hdr: str,
    bins: list,
    bin_fmt: Union[str, list[str]],
    stats_hdr: str,
    data: np.ndarray,
) -> str:
    """Prepare and print a table with data.

    Builds a histogram of ``data`` against ``bins`` and returns a
    formatted ``tabulate`` grid with four columns: the bin label, the
    count in the bin, its percentage of the total, and the cumulative
    percentage. ``bin_fmt`` controls how bin labels are rendered: a
    format string (e.g. ``".1e"``) produces ``"lo ≤ # < hi"`` labels
    formatted accordingly, while a list of strings uses each entry as a
    literal label for the corresponding bin.
    """
    # Calculate the histograms...
    hist, bins_ = np.histogram(data, bins=bins)
    bin_prcnt = 100 * hist / np.sum(hist)
    bin_cumsum_prcnt = 100 * np.cumsum(hist) / np.sum(hist)

    # Headers...
    prcnt_hdr = "% of\ntotal datasets"
    cumcum_prcnt_hdr = "cusum % of\ntotal datasets"

    tablefmt = "grid"
    if isinstance(bin_fmt, list):
        return tabulate(
            {
                bin_hdr: bin_fmt,
                stats_hdr: hist,
                prcnt_hdr: np.round(bin_prcnt, decimals=2),
                cumcum_prcnt_hdr: np.round(bin_cumsum_prcnt, decimals=2),
            },
            headers="keys",
            tablefmt=tablefmt,
        )
    else:
        return tabulate(
            {
                bin_hdr: [
                    f"{bins_[i]:{bin_fmt}} ≤ # < {bins[i + 1]:{bin_fmt}}"
                    for i in range(len(bins_) - 1)
                ],
                stats_hdr: hist,
                prcnt_hdr: np.round(bin_prcnt, decimals=2),
                cumcum_prcnt_hdr: np.round(bin_cumsum_prcnt, decimals=2),
            },
            headers="keys",
            tablefmt=tablefmt,
        )


# ---------------------------------------------------------------------------- #

cli = get_cli_args()
logging.basicConfig(level=logging.INFO if cli.verbose else logging.WARNING)

if cli.h5file.startswith(("https://", "s3://")):
    driver = "ros3"
    page_buf_size = cli.page_buffer_size
    if HDF5_VERSION < (2, 0, 0):
        s3params = get_s3_params(need_region=cli.h5file.startswith("s3://"))
    else:
        # Let the ros3 driver figure it out...
        s3params = dict()
else:
    driver = None
    page_buf_size = 0
    s3params = dict()

dset_info: list[Union[ChunkStats, ContStats]] = list()
with h5py.File(cli.h5file, mode="r", driver=driver, **s3params) as f:
    fcpl = f.id.get_create_plist()
    page = fcpl.get_file_space_strategy()[0] == h5py.h5f.FSPACE_STRATEGY_PAGE
    if page:
        page_size = fcpl.get_file_space_page_size()
    else:
        f.visititems(partial(dset_stats, dset_list=dset_info, page_size=0))

if page and page_size:
    with h5py.File(
        cli.h5file, mode="r", driver=driver, page_buf_size=page_buf_size, **s3params
    ) as f:
        f.visititems(partial(dset_stats, dset_list=dset_info, page_size=page_size))

if cli.show:
    if cli.json:
        print(json.dumps([_.to_dict() for _ in sorted(dset_info, key=lambda d: d.name)]))
    else:
        for _ in sorted(dset_info, key=lambda d: d.name):
            print(_.to_print())
    raise SystemExit()
elif cli.page_list:
    if not page:
        raise SystemExit("This option only available for a paged file.")
    if cli.page_list == "page":
        fp = {"in": {}}
        for d in dset_info:
            what = "partial" if len(d.page_bins) > 1 else "complete"
            try:
                # For out-of-pages chunked dataset...
                if d.max_size > page_size:
                    fp.setdefault("out", []).append(d.name)
                    if what == "complete":
                        continue
            except AttributeError:
                try:
                    # For out-of-pages contiguous dataset...
                    if d.out_of_page:
                        fp.setdefault("out", []).append(d.name)
                        continue
                except AttributeError:
                    pass
            for n, v in d.page_bins.items():
                fp["in"].setdefault(f"{n:04d}", {}).setdefault(what, {}).update({d.name: v})
        print(json.dumps(fp, sort_keys=True, indent=1))
    elif cli.page_list == "dataset":
        dp = dict()
        for d in dset_info:
            dp.update({d.name: {"file pages": d.page_bins}})
            try:
                dp[d.name]["number of stored chunks"] = d.num_stored
                if d.max_size > page_size:
                    dp[d.name]["out of page"] = True
                else:
                    dp[d.name]["out of page"] = False
            except AttributeError:
                dp[d.name]["out of page"] = d.out_of_page
        print(json.dumps(dp, sort_keys=True, indent=1))
    raise SystemExit()

# Split dataset info into chunked and contiguous...
cont_info: list[ContStats] = list()
chunked_info: list[ChunkStats] = list()
for _ in dset_info:
    if isinstance(_, ChunkStats):
        chunked_info.append(_)
    else:
        cont_info.append(_)
del dset_info

print(f"\nDataset statistics for {cli.h5file}")
print("Compact datasets in the file, if they exist, are excluded.")
print(f"Contiguous datasets in the file: {len(cont_info)}")
print(f"Chunked datasets in the file: {len(chunked_info)}")
if page:
    print(f'"PAGE" file space strategy with page size of {page_size:,} bytes.')
print("\n")

if cont_info:
    print(
        stats_table(
            "Contiguous dataset size\nin bytes",
            [0, 1, 1000, 10000, 100_000, 1_000_000, 4_000_000, 8_000_000, 16_000_000, np.inf],
            ".1e",
            "# contiguous\ndatasets",
            [_.stor_size for _ in cont_info],
        ),
        end="\n\n\n",
    )

    if page:
        print(
            stats_table(
                "Contiguous dataset",
                [0, 1, 2],
                ["In a file page", "Out of file pages"],
                "# contiguous\ndatasets",
                [int(_.out_of_page) for _ in cont_info],
            ),
            end="\n\n\n",
        )


if chunked_info:
    print(
        stats_table(
            "Chunked dataset total\nstored size in bytes",
            [0, 1_000_000, 4_000_000, 8_000_000, 16_000_000, 32_000_000, 64_000_000, np.inf],
            ".1e",
            "# chunked\ndatasets",
            [_.stor_size for _ in chunked_info],
        ),
        end="\n\n\n",
    )

    print(
        stats_table(
            "Chunk min size in bytes",
            [0, 1, 1000, 10000, 100_000, 1_000_000, 4_000_000, 8_000_000, 16_000_000, np.inf],
            ".1e",
            "# chunked\ndatasets",
            [_.min_size for _ in chunked_info],
        ),
        end="\n\n\n",
    )

    print(
        stats_table(
            "Chunk max size in bytes",
            [0, 1, 1000, 10000, 100_000, 1_000_000, 4_000_000, 8_000_000, 16_000_000, np.inf],
            ".1e",
            "# chunked\ndatasets",
            [_.max_size for _ in chunked_info],
        ),
        end="\n\n\n",
    )

    print(
        stats_table(
            "Chunk to dataset\nshape ratio",
            [
                0,
                0.001,
                0.002,
                0.003,
                0.004,
                0.005,
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.1,
                0.25,
                1,
            ],
            ".3f",
            "# chunked\ndatasets",
            [_.extent_ratio for _ in chunked_info],
        ),
        end="\n\n\n",
    )

    print(
        stats_table(
            "Chunks stored",
            [0, 1, 2, 10, 100, 1000, 10000, 100_000, np.inf],
            [
                "No chunks",
                "1 chunk",
                "2-9 chunks",
                "10-99 chunks",
                "100-999 chunks",
                "1000-9999 chunks",
                "10,000-99,999 chunks",
                "100,000 or more chunks",
            ],
            "# chunked\ndatasets",
            [_.num_stored for _ in chunked_info],
        ),
        end="\n\n\n",
    )

    print(
        stats_table(
            "Chunk cache size",
            [0, 1 * MiB, 4 * MiB, 8 * MiB, 16 * MiB, np.inf],
            ["1 MiB", "4 MiB", "8 MiB", "16 MiB", "> 16 MiB"],
            "# chunked\ndatasets",
            [_.size * _.num_stored for _ in chunked_info],
        ),
        end="\n\n\n" if page else "\n",
    )

    if page:
        print(
            stats_table(
                "Chunk size vs file page size",
                [0, 1, 2],
                ["All chunks in file pages", "Some chunks out of file pages"],
                "# chunked\ndatasets",
                [1 if _.max_size > page_size else 0 for _ in chunked_info],
            ),
            end="\n\n\n",
        )

        # Remove all chunked datasets with chunks bigger than one file page
        # because they are going to mess up the following stats...
        cleaned_chunked_info = [_ for _ in chunked_info if _.max_size <= page_size]
        if len(cleaned_chunked_info) < len(chunked_info):
            print(
                f"*** Removed {len(chunked_info) - len(cleaned_chunked_info)} chunked datasets "
                "with chunks stored outside of file pages because ***\n*** they are not applicable "
                "to following stats. ***",
                end="\n\n\n",
            )
            chunked_info = cleaned_chunked_info
            del cleaned_chunked_info

        print(
            stats_table(
                "# of file pages\nholding all chunks",
                [1, 2, 3, 4, 5, 6, 10, 15, 20, 25, 30, np.inf],
                [
                    "1 page",
                    "2 pages",
                    "3 pages",
                    "4 pages",
                    "5 pages",
                    "6 - 9 pages",
                    "10 - 14 pages",
                    "15 - 19 pages",
                    "20 - 24 pages",
                    "25 - 29 pages",
                    "30 or more pages",
                ],
                "# chunked\ndatasets",
                [len(_.page_bins) for _ in chunked_info],
            ),
            end="\n\n\n",
        )

        print(
            stats_table(
                "# file pages anomaly",
                [0, 1, 2, 3, 4, 5, np.inf],
                [
                    "No extra file pages",
                    "1 extra file page",
                    "2 extra file pages",
                    "3 extra file pages",
                    "4 extra file pages",
                    "5 or more extra file pages",
                ],
                "# chunked\ndatasets",
                [_.page_spread_anomaly for _ in chunked_info],
            ),
            end="\n\n\n",
        )

        print(
            stats_table(
                "Max % of chunks\nin one file page",
                [0, 20, 40, 60, 80, 100],
                ".0f",
                "# chunked\ndatasets",
                [
                    max(map(lambda x: 100 * x / _.num_stored, _.page_bins.values()))
                    for _ in chunked_info
                ],
            ),
        )