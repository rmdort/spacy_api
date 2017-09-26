FROM alpine

ENV PATH="/root/miniconda3/bin:${PATH}"

ADD https://github.com/sgerrand/alpine-pkg-glibc/releases/download/2.25-r0/glibc-2.25-r0.apk glibc.apk
ADD https://github.com/sgerrand/alpine-pkg-glibc/releases/download/2.25-r0/glibc-bin-2.25-r0.apk glibc-bin.apk
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda.sh

RUN apk add --update --no-cache --allow-untrusted git bash gcc musl-dev glibc.apk glibc-bin.apk \
  && rm glibc.apk glibc-bin.apk \
  && rm /bin/sh && ln -s /bin/bash /bin/sh

RUN chmod +x miniconda.sh \
  && ./miniconda.sh -b \
  && conda config --set add_pip_as_python_dependency false \
  && rm miniconda.sh \
  && conda update --yes conda

RUN conda config --add channels conda-forge
RUN conda install spacy
WORKDIR /app
COPY config.yml /app/
RUN pip install mprpc
RUN pip install pyyaml
RUN python -m spacy download en
RUN pip install -e git+https://github.com/rmdort/spacy_api.git@b8e66532a94e1aa45fcc3f4c88490c6604197366#egg=spacy_api
CMD ["spacy_serve", "/app/config.yml"]