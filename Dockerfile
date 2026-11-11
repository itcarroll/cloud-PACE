ARG KERNEL_ENV=repo
FROM quay.io/jupyter/minimal-notebook:2026-11-10

COPY --chown=0:0 <<EOF /etc/gitconfig
[credential]
        helper = cache --timeout 86400 --socket /tmp/git-credential-cache/socket
EOF

RUN conda install --yes jupytext

ARG KERNEL_ENV
ENV JUPYTER_DATA_DIR="${CONDA_DIR}/envs/${KERNEL_ENV}/share/jupyter/"
ADD conda-lock.yml .
RUN conda create --yes --name ${KERNEL_ENV} --file conda-lock.yml \
 && conda clean --yes --all --force
