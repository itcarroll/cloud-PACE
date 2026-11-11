FROM quay.io/jupyter/minimal-notebook:2026-11-10

ADD conda-lock.yml .
RUN conda create --yes --name env --file conda-lock.yml \
 && conda clean --yes --all --force
