FROM alpine

ENV PATH="/root/miniconda3/bin:${PATH}"

ADD https://github.com/sgerrand/alpine-pkg-glibc/releases/download/2.25-r0/glibc-2.25-r0.apk glibc.apk
ADD https://github.com/sgerrand/alpine-pkg-glibc/releases/download/2.25-r0/glibc-bin-2.25-r0.apk glibc-bin.apk
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda.sh

RUN apk add --update --no-cache --allow-untrusted bash gcc musl-dev glibc.apk glibc-bin.apk \
	&& rm glibc.apk glibc-bin.apk \
	&& rm /bin/sh && ln -s /bin/bash /bin/sh

RUN chmod +x miniconda.sh \
	&& ./miniconda.sh -b \
	&& conda config --set add_pip_as_python_dependency false \
	&& rm miniconda.sh \
	&& conda update --yes conda

WORKDIR /app
COPY . .

RUN conda env create --file requirements.yml \
	&& rm -rf /root/miniconda3/envs/spacy_api/lib/python3.6/site-packages/spacy/data \
	&& ln -s /app/models /root/miniconda3/envs/spacy_api/lib/python3.6/site-packages/spacy/data \
	&& source activate spacy_api && pip install -e . && source deactivate \
	&& conda clean --all

EXPOSE 9033 9034 9035 9036

VOLUME /app/models

CMD ["bash"]

