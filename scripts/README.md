---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: Bash
  language: bash
  name: bash
---

# scripts/README

As noted in the README, this guide is [MyST Markdown], with cells you can run interatively using the bash kernel when the [Jupytext] extension is available.

To reproduce the published results, follow the [Setup](#setup) instructions,
and then run cells below (as needed) within the same session using the activated "workspace".

[MyST Markdown]: https://mystmd.org/
[Jupytext]: https://jupytext.readthedocs.io/

## Setup

+++

Create the conda environment.

```{code-cell}
conda-lock install --name workspace ../conda-lock.yml
```

Activate the environment using your preferred distribution of conda.

+++

Activate with [Conda]:

[Conda]: https://conda-forge.org/download/

```{code-cell}
eval "$(conda shell.bash hook)"
```

```{code-cell}
conda activate workspace
```

Activate with [Micromamba]:

[Micromamba]: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html

```{code-cell}
eval "$(micromamba shell hook --shell bash)"
```

```{code-cell}
micromamba activate workspace
```

For execution on HPC, customize the following path to filesystem storage.

```{code-cell}
export DATADIR=${PWD}
```

## Reprocess FLDAS and NLDAS for Object Stores

The `reprocess` script implements cloud optimization strategies:

- Enlarge "chunks" to a size better for cloud object stores
- Move the internal data on file structure into distinct "pages"
- Copy the internal data on file structure to external "sidecar" files

Execute file reprocessing on openscapes.2i2c.cloud, using "~28 GB RAM, ~4 CPUs"

```{code-cell}
python reprocess.py --count=-1
```

Execute file reprocessing on an HPC using filesystem storage.

```{code-cell}
python reprocess.py --remote=$DATADIR --count=-1
```

## Time Data Access for Spatial Averaging

The `benchmark` script opens all files to calculate a time series of spatial averages.

Execute benchmarking on openscapes.2i2c.cloud, using "~15 GB RAM, ~1.8 CPUs".

```{code-cell}
python benchmark.py --count=-1
```

Execute benchmarking on an HPC using filesystem storage.

```{code-cell}
python benchmark.py --remote=$DATADIR --count=-1
```
