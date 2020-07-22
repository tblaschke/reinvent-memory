FROM continuumio/miniconda3:4.3.27

# Install packages to run pycharm within the container (using singularity and a local installtion in your home)
RUN apt-get update && apt-get install -y \
    git openssh-client \
    libxtst-dev libxext-dev libxrender-dev libfreetype6-dev \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r myuser && useradd -r -g myuser myuser

WORKDIR /app

# Install requirements
COPY environment.yml /app/environment.yml

RUN conda-env create -n app -f /app/environment.yml  && rm -rf /opt/conda/pkgs/*
RUN chown -R myuser:myuser /app/*

# activate the myapp environment
ENV PATH /opt/conda/envs/app/bin:$PATH