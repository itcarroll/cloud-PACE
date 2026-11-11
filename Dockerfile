FROM quay.io/jupyter/minimal-notebook:2026-11-10

ENV XDG_CACHE_HOME=/opt/conda/cache
COPY --chown=0:0 <<EOF /etc/gitconfig
[credential]
        helper = cache --timeout 86400
EOF

RUN conda install --yes jupytext \
 && conda clean --yes --all --force

ARG KERNEL_ENV=repo
ENV JUPYTER_DATA_DIR="${CONDA_DIR}/envs/${KERNEL_ENV}/share/jupyter/"
ADD conda-lock.yml .
RUN conda create --yes --name ${KERNEL_ENV} --file conda-lock.yml \
 && conda clean --yes --all --force
