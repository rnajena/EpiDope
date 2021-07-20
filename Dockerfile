FROM continuumio/miniconda3
ENV VERSION 0.3
ENV TOOL epidope

RUN apt update && apt install -y make g++ procps wget gzip && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN conda config --add channels conda-forge && \
    conda config --add channels flomock && \
    conda config --add channels default

RUN conda install mamba
RUN mamba install -y python=3.6 $TOOL=$VERSION
RUN mamba clean -ay

ENTRYPOINT ["epidope"]