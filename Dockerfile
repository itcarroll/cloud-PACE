ARG KERNEL_ENV=env
FROM quay.io/jupyter/minimal-notebook:2026-11-10

COPY --chown 0:0 <<-EOF /usr/local/bin/start-notebook.d/ssh-agent.sh
    eval `ssh-agent -s`
    EOF
    
ARG KERNEL_ENV
ADD conda-lock.yml .
RUN conda create --yes --name ${KERNEL_ENV} --file conda-lock.yml \
 && conda clean --yes --all --force
 
RUN jupyter kernelspec install --sys-prefix ${CONDA_DIR}/envs/${KERNEL_ENV}/share/jupyter/kernels/python3
