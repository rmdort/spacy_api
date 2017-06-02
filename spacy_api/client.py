import tqdm
import cachetools.func
import math
import numpy as np
from mprpc import RPCClient

# Globals
CLIENT_CACHE_SIZE = 5000


class SpacyClientToken():

    def __init__(self, **kwargs):
        self.attributes = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
        if "vector" in self.attributes:
            self.vector = np.array(self.vector)
            self.vector_norm = np.sqrt(self.vector.dot(self.vector))

    def __repr__(self):
        if "text" in self.attributes:
            return self.text
        else:
            return self.lemma_

    def similarity(self, other):
        if self.vector_norm == 0 or other.vector_norm == 0:
            return 0.0
        return np.dot(self.vector, other.vector) / (self.vector_norm * other.vector_norm)


class SpacyClientSentence(list):

    def __init__(self, tokens):
        self.tokens = [SpacyClientToken(**token) for token in tokens]
        self._vector = None
        self._vector_norm = None
        super(SpacyClientSentence, self).__init__(self.tokens)

    @property
    def vector(self):
        if self._vector is None:
            self._vector = np.mean([x.vector for x in self.tokens], axis=0)
        return self._vector

    @property
    def vector_norm(self):
        if self._vector_norm is None:
            self._vector_norm = np.sqrt(np.dot(self.vector, self.vector))
        return self._vector_norm

    @property
    def string(self):
        return "".join([x.string for x in self.tokens])

    def __getitem__(self, i):
        return self.tokens[0]

    def similarity(self, other):
        if self.vector_norm == 0 or other.vector_norm == 0:
            return 0.0
        return np.dot(self.vector, other.vector) / (self.vector_norm * other.vector_norm)


class SpacyClientDocument(list):

    def __init__(self, document):
        self.sents = [SpacyClientSentence(x) for x in document]
        self._iter = []
        super(SpacyClientDocument, self).__init__(self.sents)
        self._vector = None
        self._vector_norm = None

    @property
    def vector(self):
        if self._vector is None and len(self.sents) > 0:
            self._vector = np.mean([x.vector for x in self.sents], axis=0)
        return self._vector

    @property
    def vector_norm(self):
        if self._vector_norm is None:
            if self.vector is not None:
                self._vector_norm = np.sqrt(np.dot(self.vector, self.vector))
            else:
                self._vector_norm = 0
        return self._vector_norm

    @property
    def has_vector(self):
        return self.vector is not None

    @property
    def string(self):
        return "".join([x.string for x in self.sents])

    def __getitem__(self, i):
        if not self._iter:
            for sentence in self.sents:
                for token in sentence:
                    self._iter.append(token)
        return self._iter[i]

    def __iter__(self):
        for sentence in self.sents:
            for token in sentence:
                yield token

    def similarity(self, other):
        if self.vector_norm == 0 or other.vector_norm == 0:
            return 0.0
        return np.dot(self.vector, other.vector) / (self.vector_norm * other.vector_norm)


class BaseClient():

    def __init__(self, model, embeddings_path):
        self.model = model
        self.embeddings_path = embeddings_path

    def single(self, document, attributes):
        raise NotImplementedError

    def bulk(self, documents, attributes):
        raise NotImplementedError

    def vectors_length(self):
        raise NotImplementedError

    def description(self):
        raise NotImplementedError


class Client(BaseClient):

    def __init__(self, host="127.0.0.1", port=9033, model="en", embeddings_path=None, verbose=False):
        super(Client, self).__init__(model, embeddings_path)
        self.host = host
        self.port = port
        self.rpc = RPCClient(host, port)
        self.verbose = verbose

    def _call(self, path, *args):
        return self.rpc.call(path, *args)

    def vectors_length(self):
        length = self._call("vectors_length", self.model, self.embeddings_path)
        return int(length)

    def description(self):
        return self._call("description", self.model, self.embeddings_path)

    @cachetools.func.lru_cache(maxsize=CLIENT_CACHE_SIZE)
    def single(self, document, attributes=None):
        sentences = self._call("single", document, self.model, self.embeddings_path, attributes)
        return SpacyClientDocument(sentences)

    def _bulk(self, documents, attributes):
        return self._call("bulk", documents, self.model, self.embeddings_path, attributes)

    def bulk(self, documents, batch_size=1000, attributes=None):
        parsed_documents = []
        if len(documents) > batch_size:
            batches = int(math.ceil(len(documents) / batch_size))
            print("Batching {} requests with batch_size {}".format(batches, batch_size))
            if self.verbose:
                batch_iterator = tqdm.tqdm(range(batches))
            else:
                batch_iterator = range(batches)
            for b in batch_iterator:
                docs = documents[b * batch_size:(b + 1) * batch_size]
                res = self._bulk(docs, attributes)
                parsed_documents.extend(res)
        else:
            parsed_documents = self._bulk(documents, attributes)
        return [SpacyClientDocument(x) for x in parsed_documents]
