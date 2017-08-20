## spacy_api

Based on https://github.com/kootenpv/spacy_api

Helps with loading models in a separate, dedicated process.

Caching happens on unique arguments.

### Run it as docker container

````
docker run -p 127.0.0.1:9033:9033 registry.gitlab.com/olasearch/docker-spacy-api:v1
````

#### Features

- ✓ Serve models separately
- ✓ Client- and Server-side caching
- ✓ CLI interface

### Install spacy_api python module

Should work with py2 and py3.

````
pip install -e git+https://github.com/rmdort/spacy_api.git#egg=spacy_api
````

### Connecting to spacy
````
from spacy_api import Client

# If using kubernetes, spacy will be launched under `spacy-service-en` name
# Get Spacy server information
if os.environ.get('GET_HOSTS_FROM') == 'dns':
  spacy_host = 'spacy-service-en'
  spacy_port = 80
elif os.environ.get('GET_HOSTS_FROM') == 'env':
  spacy_host = os.environ['SPACY_SERVICE_EN_SERVICE_HOST']
  spacy_port = int(os.environ['SPACY_SERVICE_EN_SERVICE_PORT'])
else:
  # If running as a docker container
  spacy_host = '127.0.0.1'
  spacy_port = 9033

spacy_client = Client(model="en", host=spacy_host, port=spacy_port)
````

### Getting word vectors

````
doc = spacy_client.single(sentence, attributes="text,lemma_,pos,vector,has_vector")
````