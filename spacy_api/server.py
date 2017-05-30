from gevent.server import StreamServer
from mprpc import RPCServer
from spacy_api.api import single, bulk, vectors_length, description
from spacy_api.client import BaseClient


class SpacyServer(RPCServer):

    def single(self, document, model, embeddings_path, attributes):
        return single(document, model, embeddings_path, attributes)

    def bulk(self, documents, model, embeddings_path, attributes):
        documents = tuple(documents)
        return bulk(documents, model, embeddings_path, attributes)

    def vectors_length(self, model, embeddings_path):
        return vectors_length(model, embeddings_path)

    def description(self, model, embeddings_path):
        return description(model, embeddings_path)


class SpacyLocalServer(BaseClient):

    def __init__(self, model, embeddings_path):
        super(SpacyLocalServer, self).__init__(model, embeddings_path)

    def single(self, document, attributes=None):
        return single(document, self.model, self.embeddings_path, attributes, local=True)

    def bulk(self, documents, attributes=None):
        documents = tuple(documents)
        return bulk(documents, self.model, self.embeddings_path, attributes, local=True)

    def vectors_length(self):
        return vectors_length(self.model, self.embeddings_path)

    def description(self):
        return description(self.model, self.embeddings_path)


def serve(host="127.0.0.1", port=9033):
    print("Serving spacy_api at {}:{}".format(host, port))
    server = StreamServer((host, int(port)), SpacyServer())
    server.serve_forever()


if __name__ == "__main__":
    serve()
