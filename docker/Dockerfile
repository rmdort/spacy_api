FROM frolvlad/alpine-glibc:alpine-3.6

ENV PATH="/root/miniconda3/bin:${PATH}"

# ADD https://github.com/sgerrand/alpine-pkg-glibc/releases/download/2.25-r0/glibc-2.25-r0.apk glibc.apk
# ADD https://github.com/sgerrand/alpine-pkg-glibc/releases/download/2.25-r0/glibc-bin-2.25-r0.apk glibc-bin.apk
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda.sh

RUN apk add --update --no-cache --allow-untrusted git bash gcc musl-dev
# RUN rm glibc.apk glibc-bin.apk
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN chmod +x miniconda.sh \
  && ./miniconda.sh -b \
  && conda config --set add_pip_as_python_dependency false \
  && rm miniconda.sh \
  && conda update --yes conda

RUN conda config --add channels conda-forge
RUN conda install spacy
RUN conda install nltk
RUN python -m nltk.downloader 'punkt'
WORKDIR /app
COPY config.yml /app/
RUN pip install mprpc
RUN pip install pyyaml
RUN python -m spacy download en_core_web_md
RUN pip install -e git+https://github.com/rmdort/spacy_api.git@78f3c22b3b6e59758cf1a853e0fcd304d555e816#egg=spacy_api
CMD ["spacy_serve", "/app/config.yml"]