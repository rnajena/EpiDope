FROM continuumio/miniconda3
ENV VERSION 0.1
ENV TOOL epidope

RUN apt update && apt install -y make g++ procps wget gzip && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN conda config --add channels conda-forge && \
    conda config --add channels flomock && \
    conda config --add channels pytorch && \
    conda config --add channels default

RUN conda install -y python=3.7 pip $TOOL=$VERSION && conda clean -a
RUN pip install allennlp
RUN pip install docutils==0.15

ENTRYPOINT ["epidope"]