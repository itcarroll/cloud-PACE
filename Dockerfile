FROM quay.io/jupyter/minimal-notebook:2026-11-10

RUN conda config --append channels conda-pypi \
 && conda create --yes --name env --file conda-lock.yml \
 && conda clean --yes --all --force
